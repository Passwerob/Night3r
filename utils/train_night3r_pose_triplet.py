#!/usr/bin/env python3
"""Pose-head-only temporal triplet adaptation for continuous trajectories."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset, relative_pose_camera2_from_camera1
from losses.pose_triplet_losses import triplet_pose_loss
from train_event_dark3r import (
    build_event_student,
    cfg_get,
    configure_gradient_checkpointing,
    load_backbone,
    load_config,
    resolve_amp_config,
    resolve_data_root,
    resolve_model_name_or_path,
    safe_torch_load,
)
from training.event_teacher_student import build_event_optimizer, set_event_training_stage


def parse_args():
    parser = argparse.ArgumentParser("Train the Night3R pose head with temporal triplets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--init-event-checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--backbone-lora-checkpoint", default=None)
    parser.add_argument("--use-tiny-model", action="store_true")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


class PoseTripletDataset(Dataset):
    """Continuous in-scene triplets built from the existing paired-data loader."""

    def __init__(self, cfg, root_path, seed=17):
        dataset_cfg = cfg.get("dataset", {})
        lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
        lowlight_enabled = bool(lowlight_cfg.pop("enabled", False))
        lowlight_sources = lowlight_cfg.pop("apply_sources", None)
        self.base = DSECEventDataset(
            root_path=root_path,
            image_size=dataset_cfg.get("image_size", (512, 512)),
            pair_stride=1,
            pair_sampling="fixed",
            max_scenes=dataset_cfg.get("max_scenes", None),
            max_pairs_per_scene=dataset_cfg.get("max_pairs_per_scene", None),
            exclude_scenes=dataset_cfg.get("exclude_scenes", None),
            expected_event_channels=int(cfg_get(cfg, "event", "voxel_channels", 8)),
            allow_reverse_pairs=False,
            pose_output_convention=dataset_cfg.get("pose_output_convention", "native"),
            lowlight_enabled=lowlight_enabled,
            lowlight_sources=lowlight_sources,
            lowlight_kwargs=lowlight_cfg,
        )
        triplet_cfg = cfg.get("pose_triplet", {})
        self.min_stride = int(triplet_cfg.get("min_stride", 1))
        self.max_stride = int(triplet_cfg.get("max_stride", 3))
        if self.min_stride <= 0 or self.max_stride < self.min_stride:
            raise ValueError("pose_triplet strides must satisfy 0 < min_stride <= max_stride")
        self.entries = []
        for scene_dir, frames in self.base.scene_frames.items():
            poses = self.base.scene_poses.get(scene_dir)
            if poses is None:
                continue
            count = min(int(frames["num_frames"]), int(poses.shape[0]))
            for index0 in range(count):
                for stride in range(self.min_stride, self.max_stride + 1):
                    index2 = index0 + 2 * stride
                    if index2 < count:
                        self.entries.append((scene_dir, index0, stride))
        if not self.entries:
            raise RuntimeError("No continuous pose triplets were found")
        self.seed = int(seed)

    def __len__(self):
        return len(self.entries)

    def _load_frames(self, scene_dir, indices):
        frames = self.base.scene_frames[scene_dir]
        source = self.base._scene_source(scene_dir)
        images = torch.stack([self.base._load_image_01(frames["images"][index]) for index in indices])
        images_dark = images
        if self.base.lowlight_aug is not None and (
            self.base.lowlight_sources is None or source in self.base.lowlight_sources
        ):
            images_dark = self.base.lowlight_aug(images)
        events = torch.stack(
            [self.base._load_event_voxel(frames["events"][index]) for index in indices]
        )
        return images, images_dark, events, source

    def __getitem__(self, index):
        scene_dir, index0, stride = self.entries[index]
        index1 = index0 + stride
        index2 = index1 + stride
        images, images_dark, events, source = self._load_frames(
            scene_dir,
            (index0, index1, index2),
        )
        poses = self.base.scene_poses[scene_dir]
        target_01 = relative_pose_camera2_from_camera1(
            poses[index0], poses[index1], source=source, output_convention=self.base.pose_output_convention
        )
        target_12 = relative_pose_camera2_from_camera1(
            poses[index1], poses[index2], source=source, output_convention=self.base.pose_output_convention
        )
        target_02 = relative_pose_camera2_from_camera1(
            poses[index0], poses[index2], source=source, output_convention=self.base.pose_output_convention
        )
        return {
            "images": torch.stack((images[[0, 1]], images[[1, 2]], images[[0, 2]])),
            "images_dark": torch.stack(
                (images_dark[[0, 1]], images_dark[[1, 2]], images_dark[[0, 2]])
            ),
            "event_voxel": torch.stack((events[[0, 1]], events[[1, 2]], events[[0, 2]])),
            "relative_pose_gt": torch.stack((target_01, target_12, target_02)),
            "dataset_source": source,
            "scene": scene_dir.name,
            "triplet_indices": torch.tensor((index0, index1, index2), dtype=torch.long),
        }


def pair_batch(triplet_batch, pair_index):
    return {
        "images": triplet_batch["images"][:, pair_index],
        "images_dark": triplet_batch["images_dark"][:, pair_index],
        "event_voxel": triplet_batch["event_voxel"][:, pair_index],
        "dataset_source": triplet_batch["dataset_source"],
        "pose_only_inference": True,
    }


def save_pose_head(path, student, cfg, args, step, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_kind": "pose_head_only",
        "pose_head": {key: value.detach().cpu() for key, value in student.pose_head.state_dict().items()},
        "pose_head_config": dict(cfg.get("pose_head", {})),
        "base_checkpoint": str(Path(args.init_event_checkpoint).resolve()),
        "config": cfg,
        "step": int(step),
        "triplet_metrics": metrics,
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    if not bool(cfg.get("training", {}).get("pose_head_only", False)):
        raise RuntimeError("Triplet adaptation must set training.pose_head_only=true")

    dataset = PoseTripletDataset(cfg, resolve_data_root(args, cfg), seed=args.seed)
    training_cfg = cfg.get("training", {})
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 1))
    num_workers = args.num_workers if args.num_workers is not None else int(
        cfg.get("dataset", {}).get("num_workers", 0)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if not len(loader):
        raise RuntimeError("Triplet loader is empty after batching")

    backbone_source = resolve_model_name_or_path(args, cfg, "checkpoint")
    backbone = load_backbone(args, cfg, model_name_or_path=backbone_source)
    student = build_event_student(backbone, config=cfg).to(device)
    payload = safe_torch_load(
        args.init_event_checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="initial Night3R pose checkpoint",
    )
    state = payload.get("model", payload)
    missing, unexpected = student.load_state_dict(state, strict=False)
    non_pose_missing = [key for key in missing if not key.startswith("pose_head.")]
    if non_pose_missing or unexpected:
        raise RuntimeError(
            f"Triplet initialization mismatch: missing={non_pose_missing}, unexpected={list(unexpected)}"
        )
    if student.pose_head is None:
        raise RuntimeError("The configured student has no pose head")
    set_event_training_stage(student, stage=0, partial_finetune_blocks=0)
    configure_gradient_checkpointing(student, False)
    trainable = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("pose_head.") for name in trainable):
        raise RuntimeError(f"Expected pose-head-only adaptation, got trainable={trainable}")

    stage_cfg = training_cfg.get("stages", [{}])[0]
    optimizer = build_event_optimizer(
        student,
        lr_pose_head=float(stage_cfg.get("lr_pose_head", training_cfg.get("lr_pose_head", 1.0e-4))),
    )
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    triplet_cfg = cfg.get("pose_triplet", {})
    max_steps = args.max_steps or int(triplet_cfg.get("max_steps", 2000))
    save_every = int(triplet_cfg.get("save_every_steps", 200))
    grad_clip = float(training_cfg.get("grad_clip_norm", 1.0))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "resolved_triplet_config.json").write_text(
        json.dumps({"config": cfg, "args": vars(args), "triplets": len(dataset)}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "pose_triplet_train_start",
                "device": str(device),
                "triplets": len(dataset),
                "batch_size": batch_size,
                "max_steps": max_steps,
                "trainable": trainable,
                "save_every": save_every,
            }
        )
    )

    step = 0
    iterator = iter(loader)
    while step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_to_device(batch, device)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            predicted = []
            for pair_index in range(3):
                output = student(pair_batch(batch, pair_index))
                predicted.append(output[0]["relative_pose"].float())
            targets = batch["relative_pose_gt"].float()
            loss, detail = triplet_pose_loss(
                predicted[0],
                predicted[1],
                predicted[2],
                targets[:, 0],
                targets[:, 1],
                targets[:, 2],
                pair_weight=float(triplet_cfg.get("pair_weight", 1.0)),
                cycle_weight=float(triplet_cfg.get("cycle_weight", 0.5)),
                rotation_weight=float(cfg.get("loss", {}).get("pose_rotation_weight", 1.0)),
                rotation_chordal_weight=float(
                    cfg.get("loss", {}).get("pose_rotation_chordal_weight", 0.0)
                ),
                translation_weight=float(
                    cfg.get("loss", {}).get("pose_translation_weight", 1.0)
                ),
                min_translation_norm=float(
                    cfg.get("loss", {}).get("pose_min_translation_norm", 1.0e-3)
                ),
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite triplet loss at step {step}: {loss.detach().cpu()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.pose_head.parameters(), grad_clip)
        optimizer.step()
        step += 1
        metrics = {key: float(value.detach().cpu()) for key, value in detail.items()}
        metrics.update({"loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu())})
        if step == 1 or step % max(1, args.log_every) == 0:
            print(
                json.dumps(
                    {
                        "event": "pose_triplet_train_step",
                        "step": step,
                        "scene": batch["scene"],
                        "triplet_indices": batch["triplet_indices"].detach().cpu().tolist(),
                        **metrics,
                    }
                )
            )
        if step % save_every == 0 or step == max_steps:
            save_pose_head(save_dir / f"pose_head_step{step:08d}.pth", student, cfg, args, step, metrics)
    save_pose_head(save_dir / "last_pose_head.pth", student, cfg, args, step, metrics)
    print(json.dumps({"event": "pose_triplet_train_complete", "step": step, **metrics}))


if __name__ == "__main__":
    main()
