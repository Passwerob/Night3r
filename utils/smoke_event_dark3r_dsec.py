#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from losses.distillation_losses import feature_distillation_loss, structure_consistency_loss
from losses.event_geometry_losses import event_depth_edge_loss
from model.event_dark3r_student import EventDark3RStudent, make_view_pair_from_batch
from utils.smoke_event_dark3r import TinyDark3R


class DSECSmokeDataset(Dataset):
    def __init__(
        self,
        root,
        image_size=(64, 64),
        max_scenes=2,
        max_pairs_per_scene=2,
    ):
        self.root = Path(root)
        self.image_size = tuple(int(v) for v in image_size)
        if not self.root.is_dir():
            raise FileNotFoundError(f"DSEC root not found: {self.root}")

        scene_dirs = sorted(self.root.glob("screen-*-dsec"))
        if max_scenes is not None:
            scene_dirs = scene_dirs[: int(max_scenes)]
        if not scene_dirs:
            raise FileNotFoundError(f"No screen-*-dsec directories under {self.root}")

        self.samples = []
        for scene_dir in scene_dirs:
            images = sorted((scene_dir / "images").glob("*.png"))
            events = sorted((scene_dir / "events").glob("*.pt"))
            pair_count = min(len(images), len(events)) - 1
            if max_pairs_per_scene is not None:
                pair_count = min(pair_count, int(max_pairs_per_scene))
            for idx in range(max(pair_count, 0)):
                self.samples.append(
                    {
                        "image1": images[idx],
                        "image2": images[idx + 1],
                        "event1": events[idx],
                        "event2": events[idx + 1],
                        "sample_id": f"{scene_dir.name}:{idx:06d}",
                    }
                )
        if not self.samples:
            raise RuntimeError("DSEC smoke dataset has no valid adjacent image/event pairs")

    def __len__(self):
        return len(self.samples)

    def _resize(self, tensor, mode="bilinear"):
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=self.image_size, mode=mode, align_corners=False)
        return tensor.squeeze(0).contiguous()

    def _load_image(self, path):
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)
        return self._resize(image)

    def _load_event(self, path):
        try:
            event = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            event = torch.load(path, map_location="cpu")
        if not torch.is_tensor(event):
            raise TypeError(f"Expected event tensor in {path}, got {type(event)}")
        if event.ndim != 3:
            raise ValueError(f"Expected event tensor [C,H,W] in {path}, got {tuple(event.shape)}")
        if event.shape[0] != 8:
            raise ValueError(f"Expected 8 event voxel channels in {path}, got {event.shape[0]}")
        return self._resize(event.float())

    def __getitem__(self, idx):
        item = self.samples[idx]
        image1 = self._load_image(item["image1"])
        image2 = self._load_image(item["image2"])
        event1 = self._load_event(item["event1"])
        event2 = self._load_event(item["event2"])
        return {
            "images": torch.stack([image1, image2], dim=0),
            "event_voxel": torch.stack([event1, event2], dim=0),
            "sample_id": item["sample_id"],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test Event-Dark3R on extracted DSEC data.")
    parser.add_argument(
        "--data-root",
        default="/data/fcr/data/Event_Data/DSEC_EXTRACTED/dsec_chunk_001",
        help="Path containing screen-*-dsec directories.",
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=2)
    parser.add_argument("--max-pairs-per-scene", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def setup_device(device_arg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        return device, distributed, local_rank, dist.get_rank(), world_size

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    elif device_arg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, distributed, 0, 0, 1


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def main():
    args = parse_args()
    if args.height % 4 != 0 or args.width % 4 != 0:
        raise ValueError("height and width must be divisible by TinyDark3R patch_size=4")
    if int((args.height // 4) * (args.width // 4)) ** 0.5 % 1 != 0:
        raise ValueError("TinyDark3R smoke head requires a square token grid; use equal H/W.")

    torch.manual_seed(0)
    device, distributed, local_rank, rank, world_size = setup_device(args.device)

    dataset = DSECSmokeDataset(
        args.data_root,
        image_size=(args.height, args.width),
        max_scenes=args.max_scenes,
        max_pairs_per_scene=args.max_pairs_per_scene,
    )
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    teacher = TinyDark3R().to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    student = EventDark3RStudent(TinyDark3R(), config={"event": {"num_bins": 4}}).to(device)
    if distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        student = DDP(student, **ddp_kwargs)

    optimizer = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=args.lr)

    if rank == 0:
        print(
            "dsec_smoke_start",
            {
                "data_root": str(Path(args.data_root).resolve()),
                "samples": len(dataset),
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "device": str(device),
                "world_size": world_size,
            },
        )

    step_count = 0
    for step, batch in enumerate(loader):
        if step >= args.max_steps:
            break
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            view1, view2 = make_view_pair_from_batch(batch, prefer_dark=False)
            teacher_out = teacher(view1, view2)
        student_out = student(batch)

        loss_dict = {}
        feat_loss, loss_dict = feature_distillation_loss(
            student_out[0].get("feature", student_out[0]["desc"]),
            teacher_out[0].get("feature", teacher_out[0]["desc"]),
            loss_dict=loss_dict,
        )
        struct_loss, loss_dict = structure_consistency_loss(
            student_out, teacher_out, loss_dict=loss_dict
        )
        edge_loss, loss_dict = event_depth_edge_loss(
            student_out[0], batch["event_voxel"], loss_dict=loss_dict
        )
        anchor_loss = student_out[0]["pts3d"].mean() * 0.01
        total_loss = anchor_loss + feat_loss + struct_loss + edge_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite smoke loss on rank {rank}: {total_loss}")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        step_count += 1

        if rank == 0:
            print(
                "dsec_smoke_step",
                {
                    "step": step,
                    "sample_id": batch["sample_id"],
                    "images": tuple(batch["images"].shape),
                    "event_voxel": tuple(batch["event_voxel"].shape),
                    "event_enabled": student_out[0]["event_enabled"],
                    "event_features_shape": student_out[0]["event_features_shape"],
                    "event_gate": student_out[0]["event_gate"],
                    "loss": float(total_loss.detach().cpu()),
                    "loss_dict": loss_dict,
                },
            )

    if distributed:
        count_tensor = torch.tensor([step_count], device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        total_steps = int(count_tensor.item())
    else:
        total_steps = step_count

    if rank == 0:
        print("dsec_smoke_done", {"local_steps": step_count, "global_steps": total_steps})

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
