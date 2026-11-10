# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed scannet++
# dataset at https://github.com/scannetpp/scannetpp - non-commercial research and educational purposes
# https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf
# See datasets_preprocess/preprocess_scannetpp.py
# --------------------------------------------------------
import os.path as osp
import os
import cv2
import numpy as np
from skimage import io

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class ScanNetpp(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert self.split == "train"
        self.loaded_data = self._load_data()
        self.exposure_fraction = [0.06, 0.15]
        self.noise_level = [0.06, 0.15]
        self.gamma = 2.2
        self.black_level = 0.03125
        self.noisy_psnrs = []

    def _load_data(self):
        with np.load(osp.join(self.ROOT, "all_metadata.npz")) as data:
            self.scenes = data["scenes"]
            self.sceneids = data["sceneids"]
            self.images = data["images"]
            self.intrinsics = data["intrinsics"].astype(np.float32)
            self.trajectories = data["trajectories"].astype(np.float32)
            self.pairs = data["pairs"][:, :2].astype(int)

    def __len__(self):
        return len(self.pairs)

    def make_lowlight(
        self,
        image,
        verbose=False,
        debug_save_path=None,
    ):
        ungamma = (image / 255) ** (self.gamma)

        exposure_fraction = np.random.uniform(
            self.exposure_fraction[0], self.exposure_fraction[1]
        )
        scaled = ungamma * exposure_fraction

        scaled = np.clip((scaled - self.black_level) / (1 - self.black_level), 0, 1)

        noise_std = np.random.uniform(
            max(self.noise_level[0], exposure_fraction * 0.8), self.noise_level[1]
        )
        noisy = scaled + np.random.normal(0, noise_std, scaled.shape)
        # noisy /= exposure_fraction # / 2 # undo scaling
        noisy = (
            noisy / noisy.mean() * np.random.uniform(0.22, 0.26)
        )  # target mean 0.22 - 0.26
        clipped = np.clip(noisy, 0, 1)
        gamma_scaled = clipped ** (1 / self.gamma)
        gamma_scaled = (255 * gamma_scaled).astype(np.uint8)
        if verbose:
            print(
                f"Image Mean. Original: {image.mean():.2f} | Noisy: {noisy.mean():.2f} | Gamma Lowlight: {gamma_scaled.mean():.2f}"
            )
        if debug_save_path:
            os.makedirs(osp.dirname(debug_save_path), exist_ok=True)
            debug_save_path = debug_save_path.replace(
                ".jpg", f"_std{noise_std:.3f}_et{exposure_fraction:.3f}.jpg"
            )
            io.imsave(debug_save_path, gamma_scaled)
        return gamma_scaled

    def _get_views(self, idx, resolution, rng):

        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        for view_idx in [image_idx1, image_idx2]:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, "images", basename + ".jpg"))
            if (
                idx % 100 == 0
            ):  # TODO: Change this path to be in the log dir for the specific experiment
                noisy_image = self.make_lowlight(
                    rgb_image,
                    debug_save_path=f"scratch/scannetpp_lowlight_test/{str(idx).zfill(6)}_{str(view_idx).zfill(6)}.jpg",
                )
            else:
                noisy_image = self.make_lowlight(rgb_image)

            # DELETE LATER - WILL SLOW THIS DOWN
            # compute psnr of noisy vs rgb_image
            noisy_psnr = cv2.PSNR(rgb_image, noisy_image)
            self.noisy_psnrs.append(noisy_psnr)
            if idx % 20 == 0:  # DELETE LATER
                print(
                    f"sample size: {len(self.noisy_psnrs)} | mean noisy psnr: {np.mean(self.noisy_psnrs):.2f} "
                )
            ###
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED
            )

            # QUESTION: how to best add noise to depth? Should the variance be the same? Should it be different? is the depthmap used to supervise? Mast3r should not get it in the beginning, so might not need to add noise to the ddepthmap
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            noisy_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                noisy_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=noisy_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="ScanNet++",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                )
            )
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = ScanNetpp(
        split="train", ROOT="data/scannetpp_processed", resolution=224, aug_crop=16
    )

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]["camera_pose"] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]["pts3d"]
            valid_mask = views[view_idx]["valid_mask"]
            colors = rgb(views[view_idx]["img"])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(
                pose_c2w=views[view_idx]["camera_pose"],
                focal=views[view_idx]["camera_intrinsics"][0, 0],
                color=(idx * 255, (1 - idx) * 255, 0),
                image=colors,
                cam_size=cam_size,
            )
        viz.show()
