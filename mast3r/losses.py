# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of MASt3R training losses
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.losses import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from dust3r.losses import Regr3D as Regr3D_dust3r
from dust3r.utils.geometry import geotrf, inv, normalize_pointcloud
from dust3r.inference import get_pred_pts3d
from dust3r.utils.geometry import (
    get_joint_pointcloud_depth,
    get_joint_pointcloud_center_scale,
)


def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz


class Regr3D(Regr3D_dust3r):
    def __init__(
        self,
        criterion,
        norm_mode="avg_dis",
        gt_scale=False,
        opt_fit_gt=False,
        sky_loss_value=2,
        max_metric_scale=False,
        loss_in_log=False,
    ):
        self.loss_in_log = loss_in_log
        if norm_mode.startswith("?"):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        super().__init__(criterion, self.norm_mode, gt_scale)

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1["camera_pose"])
        gt_pts1 = geotrf(in_camera1, gt1["pts3d"])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2["pts3d"])  # B,H,W,3

        valid1 = gt1["valid_mask"].clone()
        valid2 = gt2["valid_mask"].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        if self.loss_in_log == "before":
            # this only make sense when depth_mode == 'linear'
            gt_pts1 = apply_log_to_norm(gt_pts1)
            gt_pts2 = apply_log_to_norm(gt_pts2)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()

        if not self.norm_all:
            if self.max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(
                    valid1, torch.linalg.norm(gt_pts1, dim=-1), 0
                ).view(B, -1)
                dist2_to_cam1 = torch.where(
                    valid2, torch.linalg.norm(gt_pts2, dim=-1), 0
                ).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1["is_metric_scale"] = (
                    gt1["is_metric_scale"]
                    & (dist1_to_cam1.max(dim=-1).values < self.max_metric_scale)
                    & (dist2_to_cam1.max(dim=-1).values < self.max_metric_scale)
                )
                gt2["is_metric_scale"] = gt1["is_metric_scale"]

            mask = ~gt1["is_metric_scale"]
        else:
            mask = torch.ones_like(gt1["is_metric_scale"])
        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(
                pr_pts1[mask], pr_pts2[mask], self.norm_mode, valid1[mask], valid2[mask]
            )

        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2, norm_factor = normalize_pointcloud(
                gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, ret_factor=True
            )
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1["sky_mask"] & (~valid1)
        sky2 = gt2["sky_mask"] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        (
            gt_pts1,
            gt_pts2,
            pred_pts1,
            pred_pts2,
            mask1,
            mask2,
            sky1,
            sky2,
            monitoring,
        ) = self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            # add the sky pixel as "valid" pixels...
            mask1 = mask1 | sky1
            mask2 = mask2 | sky2

        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        if self.loss_in_log and self.loss_in_log != "before":
            # this only make sense when depth_mode == 'exp'
            pred_pts1 = apply_log_to_norm(pred_pts1)
            gt_pts1 = apply_log_to_norm(gt_pts1)
        l1 = self.criterion(pred_pts1, gt_pts1)

        # loss on gt2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        if self.loss_in_log and self.loss_in_log != "before":
            pred_pts2 = apply_log_to_norm(pred_pts2)
            gt_pts2 = apply_log_to_norm(gt_pts2)
        l2 = self.criterion(pred_pts2, gt_pts2)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            # ... but force the loss to be high there
            l1 = torch.where(sky1[mask1], self.sky_loss_value, l1)
            l2 = torch.where(sky2[mask2], self.sky_loss_value, l2)
        self_name = type(self).__name__
        details = {
            self_name + "_pts3d_1": float(l1.mean()),
            self_name + "_pts3d_2": float(l2.mean()),
        }
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class Regr3D_ShiftInv(Regr3D):
    """Same than Regr3D but invariant to depth shift."""

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        (
            gt_pts1,
            gt_pts2,
            pred_pts1,
            pred_pts2,
            mask1,
            mask2,
            sky1,
            sky2,
            monitoring,
        ) = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[
            :, None, None
        ]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[
            :, None, None
        ]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return (
            gt_pts1,
            gt_pts2,
            pred_pts1,
            pred_pts2,
            mask1,
            mask2,
            sky1,
            sky2,
            monitoring,
        )


class Regr3D_ScaleInv(Regr3D):
    """Same than Regr3D but invariant to depth scale.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        (
            gt_pts1,
            gt_pts2,
            pred_pts1,
            pred_pts2,
            mask1,
            mask2,
            sky1,
            sky2,
            monitoring,
        ) = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(
            pred_pts1, pred_pts2, mask1, mask2
        )

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return (
            gt_pts1,
            gt_pts2,
            pred_pts1,
            pred_pts2,
            mask1,
            mask2,
            sky1,
            sky2,
            monitoring,
        )


class Regr3D_ScaleShiftInv(Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass


def get_similarities(desc1, desc2, euc=False):
    if euc:  # euclidean distance in same range than similarities
        dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        sim = 1 / (1 + dists)
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim


class MatchingCriterion(BaseCriterion):
    def __init__(self, reduction="mean", fp=torch.float32):
        super().__init__(reduction)
        self.fp = fp

    def forward(self, a, b, valid_matches=None, euc=False):
        assert a.ndim >= 2 and 1 <= a.shape[-1], f"Bad shape = {a.shape}"
        dist = self.loss(a.to(self.fp), b.to(self.fp), valid_matches, euc=euc)
        # one dimension less or reduction to single value
        assert (
            valid_matches is None and dist.ndim == a.ndim - 1
        ) or self.reduction in ["mean", "sum", "1-mean", "none"]
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        if self.reduction == "1-mean":
            return 1.0 - dist.mean() if dist.numel() > 0 else dist.new_ones(())
        raise ValueError(f"bad {self.reduction=} mode")

    def loss(self, a, b, valid_matches=None):
        raise NotImplementedError


class InfoNCE(MatchingCriterion):
    def __init__(self, temperature=0.07, eps=1e-8, mode="all", **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.eps = eps
        assert mode in ["all", "proper", "dual"]
        self.mode = mode

    def loss(self, desc1, desc2, valid_matches=None, euc=False):
        # valid positives are along diagonals
        B, N, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2
        if valid_matches is None:
            valid_matches = torch.ones([B, N], dtype=bool)
        # torch.all(valid_matches.sum(dim=-1) > 0) some pairs have no matches????
        if valid_matches.sum() == 0:
            print(
                f"Warning. No valid matches found in InfoNCE loss, returning large penalty."
            )
            penalty_tensor = torch.tensor(
                [0.0], device=desc1.device, dtype=torch.float32
            )
            return penalty_tensor  # no valid matches, return a large loss
        assert valid_matches.shape == torch.Size([B, N]) and valid_matches.sum() > 0

        # Tempered similarities
        sim = get_similarities(desc1, desc2, euc) / self.temperature
        sim[sim.isnan()] = -torch.inf  # ignore nans
        # Softmax of positives with temperature
        sim = sim.exp_()  # save peak memory
        positives = sim.diagonal(dim1=-2, dim2=-1)

        # Loss
        if self.mode == "all":  # Previous InfoNCE
            loss = -torch.log(
                (positives / sim.sum(dim=-1).sum(dim=-1, keepdim=True)).clip(self.eps)
            )
        elif self.mode == "proper":  # Proper InfoNCE
            loss = -(
                torch.log((positives / sim.sum(dim=-2)).clip(self.eps))
                + torch.log((positives / sim.sum(dim=-1)).clip(self.eps))
            )
        elif self.mode == "dual":  # Dual Softmax
            loss = -(
                torch.log(
                    (positives**2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps)
                )
            )
        else:
            raise ValueError("This should not happen...")
        return loss[valid_matches]


class APLoss(MatchingCriterion):
    """AP loss"""

    def __init__(self, nq="torch", min=0, max=1, euc=False, **kw):
        super().__init__(**kw)
        # Exact/True AP loss (not differentiable)
        if nq == 0:
            nq = "sklearn"  # special case
        try:
            self.compute_AP = eval("self.compute_true_AP_" + nq)
        except:
            raise ValueError("Unknown mode %s for AP loss" % nq)

    @staticmethod
    def compute_true_AP_sklearn(scores, labels):
        def compute_AP(label, score):
            return average_precision_score(label, score)

        aps = scores.new_zeros((scores.shape[0], scores.shape[1]))
        label_np = labels.cpu().numpy().astype(bool)
        scores_np = scores.cpu().numpy()
        for bi in range(scores_np.shape[0]):
            for i in range(scores_np.shape[1]):
                labels = label_np[bi, i, :]
                if labels.sum() < 1:
                    continue
                aps[bi, i] = compute_AP(labels, scores_np[bi, i, :])
        return aps

    @staticmethod
    def compute_true_AP_torch(scores, labels):
        assert scores.shape == labels.shape
        B, N, M = labels.shape
        dev = labels.device
        with torch.no_grad():
            # sort scores
            _, order = scores.sort(dim=-1, descending=True)
            # sort labels accordingly
            labels = labels[
                torch.arange(B, device=dev)[:, None, None].expand(order.shape),
                torch.arange(N, device=dev)[None, :, None].expand(order.shape),
                order,
            ]
            # compute number of positives per query
            npos = labels.sum(dim=-1)
            assert torch.all(
                torch.isclose(npos, npos[0, 0])
            ), "only implemented for constant number of positives per query"
            npos = int(npos[0, 0])
            # compute precision at each recall point
            posrank = labels.nonzero()[:, -1].view(B, N, npos)
            recall = torch.arange(1, 1 + npos, dtype=torch.float32, device=dev)[
                None, None, :
            ].expand(B, N, npos)
            precision = recall / (1 + posrank).float()
            # average precision values at all recall points
            aps = precision.mean(dim=-1)

        return aps

    def loss(
        self, desc1, desc2, valid_matches=None, euc=False
    ):  # if matches is None, positives are the diagonal
        B, N1, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2

        scores = get_similarities(desc1, desc2, euc)

        labels = torch.zeros([B, N1, N2], dtype=scores.dtype, device=scores.device)

        # allow all diagonal positives and only mask afterwards
        labels.diagonal(dim1=-2, dim2=-1)[...] = 1.0
        apscore = self.compute_AP(scores, labels)
        if valid_matches is not None:
            apscore = apscore[valid_matches]
        return apscore
