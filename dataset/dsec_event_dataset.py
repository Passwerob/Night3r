import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset

from dataset.lowlight_augmentation import ContinuousLowLightAug


def _to_pair(value, default):
    if value is None:
        value = default
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError("image_size must be an int or a pair of ints")
    return int(value[0]), int(value[1])


def _resize_chw(tensor, image_size, mode="bilinear"):
    tensor = tensor.unsqueeze(0)
    kwargs = {"size": image_size, "mode": mode}
    if mode in ("bilinear", "bicubic", "linear", "trilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs).squeeze(0).contiguous()


def _normalize_for_dark3r(images):
    return images * 2.0 - 1.0


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _root_paths(root_path):
    if isinstance(root_path, (list, tuple)):
        values = root_path
    else:
        values = [root_path]
    paths = [Path(value) for value in values if value is not None and str(value)]
    if not paths:
        raise ValueError("At least one dataset root is required")
    return paths


def relative_pose_camera2_from_camera1(
    t_world_camera1,
    t_world_camera2,
    source="unknown",
    output_convention="native",
):
    """Return T_camera2_from_camera1, optionally in OpenCV camera axes."""
    relative_pose = torch.linalg.inv(t_world_camera2) @ t_world_camera1
    if str(source).strip().lower() == "dl3dv" and output_convention == "opencv":
        axis_flip = torch.diag(relative_pose.new_tensor([1.0, -1.0, -1.0, 1.0]))
        relative_pose = axis_flip @ relative_pose @ axis_flip
    return relative_pose


class DSECEventDataset(Dataset):
    """Event/RGB pairs exported as screen-* directories.

    Each sample returns a compact Event-Dark3R batch item:

    - images: [2,3,H,W], normalized to [-1,1] for Dark3R/MASt3R.
    - images_photo: [2,3,H,W], kept in [0,1] for photometric/event losses.
    - event_voxel: [2,8,H,W], loaded from precomputed voxel tensors.
    """

    def __init__(
        self,
        root_path,
        image_size=(512, 512),
        pair_stride=1,
        max_scenes=None,
        max_pairs_per_scene=None,
        exclude_scenes=None,
        expected_event_channels=8,
        pair_sampling="fixed",
        min_pair_distance=None,
        max_pair_distance=None,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        pose_output_convention="native",
        lowlight_enabled=False,
        lowlight_sources=None,
        lowlight_kwargs=None,
    ):
        self.root_paths = _root_paths(root_path)
        self.root_path = self.root_paths[0]
        self.root_paths_text = ", ".join(str(path) for path in self.root_paths)
        self.image_size = _to_pair(image_size, (512, 512))
        self.pair_stride = int(pair_stride)
        self.pair_sampling = str(pair_sampling or "fixed").lower()
        if precomputed_pairs is not None and self.pair_sampling == "fixed":
            self.pair_sampling = "precomputed"
        if self.pair_sampling not in {"fixed", "random", "precomputed"}:
            raise ValueError("pair_sampling must be fixed, random, or precomputed")
        self.min_pair_distance = (
            self.pair_stride if min_pair_distance is None else int(min_pair_distance)
        )
        self.max_pair_distance = (
            self.pair_stride if max_pair_distance is None else int(max_pair_distance)
        )
        if self.min_pair_distance <= 0 or self.max_pair_distance <= 0:
            raise ValueError("pair distances must be positive")
        if self.min_pair_distance > self.max_pair_distance:
            raise ValueError("min_pair_distance must be <= max_pair_distance")
        self.precomputed_pairs = precomputed_pairs
        self.allow_reverse_pairs = bool(allow_reverse_pairs)
        self.pose_output_convention = str(pose_output_convention).strip().lower()
        if self.pose_output_convention not in {"native", "opencv"}:
            raise ValueError("pose_output_convention must be 'native' or 'opencv'")
        self.lowlight_sources = (
            None
            if lowlight_sources is None
            else {str(source).strip().lower() for source in lowlight_sources}
        )
        self.max_scenes = None if max_scenes is None else int(max_scenes)
        self.max_pairs_per_scene = (
            None if max_pairs_per_scene is None else int(max_pairs_per_scene)
        )
        self.exclude_scenes = {
            str(scene).strip() for scene in (exclude_scenes or []) if str(scene).strip()
        }
        self.excluded_scenes_found = set()
        self.expected_event_channels = int(expected_event_channels)
        if self.pair_stride <= 0:
            raise ValueError("pair_stride must be positive")
        missing_roots = [path for path in self.root_paths if not path.is_dir()]
        if missing_roots:
            raise FileNotFoundError(
                "DSEC/MVSEC root not found: "
                + ", ".join(str(path) for path in missing_roots)
            )

        lowlight_kwargs = {} if lowlight_kwargs is None else dict(lowlight_kwargs)
        self.lowlight_aug = (
            ContinuousLowLightAug(**lowlight_kwargs) if lowlight_enabled else None
        )

        self.scene_camera = {}
        self.scene_meta = {}
        self.scene_poses = {}
        self.scene_frames = {}
        self.samples = []
        self.has_depth_gt = False
        self.has_all_depth_gt = True
        self.has_pseudo_depth = False
        self.has_all_pseudo_depth = True
        self.has_relative_pose = False
        self.has_all_relative_pose = True
        self._discover_samples()
        if not self.samples:
            raise RuntimeError(f"No image/event pairs found under {self.root_paths_text}")

    def _discover_samples(self):
        scene_dirs = []
        for root_path in self.root_paths:
            scene_dirs.extend(
                sorted(
                    scene_dir
                    for scene_dir in root_path.glob("screen-*")
                    if scene_dir.is_dir()
                )
            )
        if self.exclude_scenes:
            self.excluded_scenes_found = {
                scene_dir.name for scene_dir in scene_dirs if scene_dir.name in self.exclude_scenes
            }
            scene_dirs = [
                scene_dir for scene_dir in scene_dirs if scene_dir.name not in self.exclude_scenes
            ]
        if self.max_scenes is not None:
            scene_dirs = scene_dirs[: self.max_scenes]

        for scene_dir in scene_dirs:
            images = sorted((scene_dir / "images").glob("*.png"))
            events = sorted((scene_dir / "events").glob("*.pt"))
            if not images or not events:
                continue
            depth_dir = scene_dir / "depth_gt"
            depths = sorted(depth_dir.glob("*.pt")) if depth_dir.is_dir() else []
            pseudo_depth_dir = scene_dir / "depth_pseudo"
            pseudo_depths = (
                sorted(pseudo_depth_dir.glob("*.pt")) if pseudo_depth_dir.is_dir() else []
            )
            pseudo_conf_dir = scene_dir / "depth_pseudo_conf"
            pseudo_confs = (
                sorted(pseudo_conf_dir.glob("*.pt")) if pseudo_conf_dir.is_dir() else []
            )
            self.scene_meta[scene_dir] = self._load_meta(scene_dir)
            self.scene_camera[scene_dir] = self._load_scene_camera(scene_dir)
            scene_poses = self._load_scene_poses(scene_dir)
            self.scene_poses[scene_dir] = scene_poses
            n_frames = min(len(images), len(events))
            if depths:
                n_frames = min(n_frames, len(depths))
            if pseudo_depths:
                n_frames = min(n_frames, len(pseudo_depths))
                if pseudo_confs:
                    n_frames = min(n_frames, len(pseudo_confs))
            if scene_poses is not None:
                n_frames = min(n_frames, scene_poses.shape[0])
            if n_frames <= 1:
                continue
            if depths:
                self.has_depth_gt = True
            else:
                self.has_all_depth_gt = False
            if pseudo_depths:
                self.has_pseudo_depth = True
            else:
                self.has_all_pseudo_depth = False
            if scene_poses is not None:
                self.has_relative_pose = True
            else:
                self.has_all_relative_pose = False
            images = images[:n_frames]
            events = events[:n_frames]
            depths = depths[:n_frames] if depths else []
            pseudo_depths = pseudo_depths[:n_frames] if pseudo_depths else []
            pseudo_confs = pseudo_confs[:n_frames] if pseudo_confs else []
            self.scene_frames[scene_dir] = {
                "images": images,
                "events": events,
                "depths": depths,
                "pseudo_depths": pseudo_depths,
                "pseudo_confs": pseudo_confs,
                "num_frames": n_frames,
            }

            scene_samples = []
            if self.pair_sampling == "fixed":
                pair_count = n_frames - self.pair_stride
                for idx in range(max(pair_count, 0)):
                    scene_samples.append(self._make_sample(scene_dir, idx, idx + self.pair_stride))
            elif self.pair_sampling == "precomputed":
                precomputed = self._load_precomputed_pairs(scene_dir, n_frames)
                for idx in sorted(precomputed):
                    candidates = sorted(set(precomputed[idx]))
                    if candidates:
                        scene_samples.append(
                            self._make_sample(scene_dir, idx, candidates[0], candidates)
                        )
            else:
                for idx in range(n_frames):
                    candidates = self._random_pair_candidates(idx, n_frames)
                    if candidates:
                        scene_samples.append(
                            self._make_sample(scene_dir, idx, candidates[0], candidates)
                        )
            if self.max_pairs_per_scene is not None:
                scene_samples = scene_samples[: self.max_pairs_per_scene]
            self.samples.extend(scene_samples)

    def _make_sample(self, scene_dir, idx1, idx2, pair_candidates=None):
        sample = {
            "scene_dir": scene_dir,
            "scene": scene_dir.name,
            "idx1": int(idx1),
            "idx2": int(idx2),
            "pair_sampling": self.pair_sampling,
        }
        if pair_candidates is not None:
            sample["pair_candidates"] = [int(idx) for idx in pair_candidates]
        return sample

    def _random_pair_candidates(self, idx, n_frames):
        candidates = []
        for distance in range(self.min_pair_distance, self.max_pair_distance + 1):
            forward = idx + distance
            if forward < n_frames:
                candidates.append(forward)
            backward = idx - distance
            if self.allow_reverse_pairs and backward >= 0:
                candidates.append(backward)
        return candidates

    def _load_precomputed_pairs(self, scene_dir, n_frames):
        if self.precomputed_pairs is None:
            return {}
        precomputed_path = Path(self.precomputed_pairs)
        candidates = [precomputed_path] if precomputed_path.is_absolute() else []
        if not precomputed_path.is_absolute():
            candidates.extend([scene_dir / precomputed_path, scene_dir / "meta" / precomputed_path])
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            return {}
        with open(path, "r") as f:
            raw_pairs = json.load(f)

        pairs = {}

        def add_pair(i, j):
            i = int(i)
            j = int(j)
            if i == j:
                return
            if 0 <= i < n_frames and 0 <= j < n_frames:
                pairs.setdefault(i, []).append(j)

        if isinstance(raw_pairs, dict):
            for idx, values in raw_pairs.items():
                if isinstance(values, int):
                    add_pair(idx, values)
                else:
                    for value in values:
                        if isinstance(value, (list, tuple)) and len(value) >= 2:
                            add_pair(value[0], value[1])
                        else:
                            add_pair(idx, value)
        elif isinstance(raw_pairs, list):
            for item in raw_pairs:
                if isinstance(item, dict):
                    i = item.get("idx1", item.get("i", item.get("src")))
                    j = item.get("idx2", item.get("j", item.get("dst")))
                    if i is not None and j is not None:
                        add_pair(i, j)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    add_pair(item[0], item[1])
        return pairs

    def _load_scene_camera(self, scene_dir):
        camera_path = scene_dir / "cameras.yaml"
        if camera_path.is_file():
            with open(camera_path, "r") as f:
                cameras = yaml.safe_load(f)
            camera = cameras.get("rgb_cam_crop", cameras.get("rgb_cam_full", None))
            if camera is None:
                return None
            fx, fy, cx, cy = [float(x) for x in camera["intrinsics"]]
            width, height = [float(x) for x in camera["resolution"]]
        else:
            transforms_path = scene_dir / "transforms.json"
            if not transforms_path.is_file():
                return None
            transforms = _load_json(transforms_path)
            required = ("fl_x", "fl_y", "cx", "cy", "w", "h")
            if not all(key in transforms for key in required):
                return None
            fx = float(transforms["fl_x"])
            fy = float(transforms["fl_y"])
            cx = float(transforms["cx"])
            cy = float(transforms["cy"])
            width = float(transforms["w"])
            height = float(transforms["h"])
        target_h, target_w = self.image_size
        sx = target_w / width
        sy = target_h / height
        intrinsics = torch.tensor(
            [
                [fx * sx, 0.0, cx * sx],
                [0.0, fy * sy, cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return intrinsics

    def _load_scene_poses(self, scene_dir):
        poses_path = scene_dir / "meta" / "poses_selected.npz"
        if poses_path.is_file():
            poses_npz = np.load(poses_path)
            for key in ("T_world_body", "T_world_cam", "T_world_camera"):
                if key in poses_npz:
                    poses = torch.from_numpy(poses_npz[key]).float()
                    if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
                        return poses
        transforms_path = scene_dir / "transforms.json"
        if not transforms_path.is_file():
            return None
        transforms = _load_json(transforms_path)
        frames = transforms.get("frames", None)
        if not isinstance(frames, list) or not frames:
            return None
        poses = []
        for frame in frames:
            matrix = frame.get("transform_matrix", None)
            if matrix is None:
                return None
            pose = torch.as_tensor(matrix, dtype=torch.float32)
            if pose.shape != (4, 4):
                return None
            poses.append(pose)
        if not poses:
            return None
        return torch.stack(poses, dim=0)
        return None

    def __len__(self):
        return len(self.samples)

    def _load_image_01(self, path):
        image = Image.open(path).convert("RGB")
        target_h, target_w = self.image_size
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_event_voxel(self, path):
        try:
            event = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            event = torch.load(path, map_location="cpu")
        if not torch.is_tensor(event):
            raise TypeError(f"Expected tensor event voxel in {path}, got {type(event)}")
        if event.ndim != 3:
            raise ValueError(f"Expected event voxel [C,H,W] in {path}, got {event.shape}")
        if event.shape[0] != self.expected_event_channels:
            raise ValueError(
                f"Expected {self.expected_event_channels} event channels in {path}, "
                f"got {event.shape[0]}"
            )
        event = event.float()
        if tuple(event.shape[-2:]) != self.image_size:
            event = _resize_chw(event, self.image_size, mode="bilinear")
        return event

    def _load_depth_gt(self, path):
        try:
            depth = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            depth = torch.load(path, map_location="cpu")
        if not torch.is_tensor(depth):
            depth = torch.as_tensor(depth)
        depth = depth.float()
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        if depth.ndim != 2:
            raise ValueError(f"Expected depth_gt [H,W] in {path}, got {depth.shape}")
        depth = depth.unsqueeze(0)
        if tuple(depth.shape[-2:]) != self.image_size:
            depth = _resize_chw(depth, self.image_size, mode="nearest")
        return depth.squeeze(0).contiguous()

    def _load_meta(self, scene_dir):
        meta_path = scene_dir / "meta.json"
        if not meta_path.is_file():
            return {}
        with open(meta_path, "r") as f:
            return json.load(f)

    def _scene_source(self, scene_dir):
        meta = self.scene_meta.get(scene_dir, {})
        source = meta.get("source_dataset", meta.get("source", None))
        if source:
            return str(source).strip().lower()
        scene_dir_text = str(scene_dir).strip().lower()
        if "dl3dv" in scene_dir_text:
            return "dl3dv"
        for key in (
            "source_scene_dir",
            "source_images_dir",
            "source_transforms_json",
            "source_dvs_clip_dir",
            "source_dvs_video_dir",
        ):
            value = meta.get(key, None)
            if value and "dl3dv" in str(value).strip().lower():
                return "dl3dv"
        name = scene_dir.name.lower()
        if name.endswith("-mvsec"):
            return "mvsec"
        if name.endswith("-dsec"):
            return "dsec"
        return "unknown"

    def __getitem__(self, idx):
        item = self.samples[idx]
        frames = self.scene_frames[item["scene_dir"]]
        idx1 = int(item["idx1"])
        if item.get("pair_candidates"):
            idx2 = int(random.choice(item["pair_candidates"]))
        else:
            idx2 = int(item["idx2"])

        image1_path = frames["images"][idx1]
        image2_path = frames["images"][idx2]
        event1_path = frames["events"][idx1]
        event2_path = frames["events"][idx2]

        meta = self.scene_meta.get(item["scene_dir"], {})
        dataset_source = self._scene_source(item["scene_dir"])
        image1 = self._load_image_01(image1_path)
        image2 = self._load_image_01(image2_path)
        images_photo = torch.stack([image1, image2], dim=0)
        should_darken = self.lowlight_aug is not None and (
            self.lowlight_sources is None or dataset_source in self.lowlight_sources
        )
        if should_darken:
            images_photo_dark = self.lowlight_aug(images_photo)
        else:
            images_photo_dark = images_photo

        event1 = self._load_event_voxel(event1_path)
        event2 = self._load_event_voxel(event2_path)
        intrinsics = self.scene_camera.get(item["scene_dir"], None)
        if intrinsics is None:
            intrinsics = torch.full((3, 3), float("nan"), dtype=torch.float32)

        sample = {
            "images": _normalize_for_dark3r(images_photo),
            "images_dark": _normalize_for_dark3r(images_photo_dark),
            "images_photo": images_photo,
            "images_photo_dark": images_photo_dark,
            "event_voxel": torch.stack([event1, event2], dim=0),
            "intrinsics": intrinsics,
            "scene": item["scene"],
            "sample_id": f"{item['scene']}:{idx1:06d}-{idx2:06d}",
            "image1_path": str(image1_path),
            "image2_path": str(image2_path),
            "event1_path": str(event1_path),
            "event2_path": str(event2_path),
            "idx1": idx1,
            "idx2": idx2,
            "pair_sampling": item.get("pair_sampling", self.pair_sampling),
            "dataset_source": dataset_source,
            "depth_protocol": str(meta.get("depth_protocol", "")),
        }
        poses = self.scene_poses.get(item["scene_dir"], None)
        if poses is not None:
            t_world_1 = poses[idx1]
            t_world_2 = poses[idx2]
            relative_pose = relative_pose_camera2_from_camera1(
                t_world_1,
                t_world_2,
                source=dataset_source,
                output_convention=self.pose_output_convention,
            )
            sample.update(
                {
                    "relative_pose": relative_pose,
                    "relative_pose_gt": relative_pose,
                    "pose_source": (
                        "poses_selected.npz"
                        if (item["scene_dir"] / "meta" / "poses_selected.npz").is_file()
                        else "transforms.json"
                    ),
                }
            )
        if frames["depths"]:
            depth1_path = frames["depths"][idx1]
            depth2_path = frames["depths"][idx2]
            depth1 = self._load_depth_gt(depth1_path)
            depth2 = self._load_depth_gt(depth2_path)
            sample.update(
                {
                    "depth_gt": depth1,
                    "depth_gt1": depth1,
                    "depth_gt2": depth2,
                    "depth_valid": torch.isfinite(depth1) & (depth1 > 0),
                    "depth_source": str(meta.get("depth_source", "real")),
                    "depth1_path": str(depth1_path),
                    "depth2_path": str(depth2_path),
                }
            )
        if frames["pseudo_depths"]:
            pseudo_depth1_path = frames["pseudo_depths"][idx1]
            pseudo_depth2_path = frames["pseudo_depths"][idx2]
            pseudo_depth1 = self._load_depth_gt(pseudo_depth1_path)
            pseudo_depth2 = self._load_depth_gt(pseudo_depth2_path)
            pseudo_sample = {
                "depth_pseudo": pseudo_depth1,
                "depth_pseudo1": pseudo_depth1,
                "depth_pseudo2": pseudo_depth2,
                "depth_pseudo_valid": (
                    torch.isfinite(pseudo_depth1) & (pseudo_depth1 > 0)
                ),
                "depth_pseudo_source": str(
                    meta.get("pseudo_depth_source", "external_teacher")
                ),
                "depth_pseudo1_path": str(pseudo_depth1_path),
                "depth_pseudo2_path": str(pseudo_depth2_path),
            }
            if frames["pseudo_confs"]:
                pseudo_conf1_path = frames["pseudo_confs"][idx1]
                pseudo_conf2_path = frames["pseudo_confs"][idx2]
                pseudo_sample.update(
                    {
                        "depth_pseudo_conf": self._load_depth_gt(pseudo_conf1_path),
                        "depth_pseudo_conf1": self._load_depth_gt(pseudo_conf1_path),
                        "depth_pseudo_conf2": self._load_depth_gt(pseudo_conf2_path),
                        "depth_pseudo_conf1_path": str(pseudo_conf1_path),
                        "depth_pseudo_conf2_path": str(pseudo_conf2_path),
                    }
                )
            sample.update(pseudo_sample)
        return sample
