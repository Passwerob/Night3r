import torch


def compute_regr_loss(
    clean_orig_res11,
    clean_orig_res21,
    clean_orig_res22,
    clean_orig_res12,
    lowlight_res11,
    lowlight_res21,
    lowlight_res22,
    lowlight_res12,
):
    """Compute regression loss for 3D points and confidence."""
    gt_pts3d = [
        clean_orig_res11["pts3d"],
        clean_orig_res21["pts3d"],
        clean_orig_res22["pts3d"],
        clean_orig_res12["pts3d"],
    ]

    pred_pts3d = [
        lowlight_res11["pts3d"],
        lowlight_res21["pts3d"],
        lowlight_res22["pts3d"],
        lowlight_res12["pts3d"],
    ]
    pred_conf = [
        lowlight_res11["conf"],
        lowlight_res21["conf"],
        lowlight_res22["conf"],
        lowlight_res12["conf"],
    ]

    loss = torch.zeros((), device=lowlight_res11["pts3d"].device)
    conf_loss = torch.zeros((), device=lowlight_res11["pts3d"].device)

    for gt, pred, conf in zip(gt_pts3d, pred_pts3d, pred_conf):
        distance = torch.norm(gt - pred, dim=-1)
        loss += (distance * conf).mean()  # see eq. 7 in Mast3r paper
        conf_loss -= torch.log(conf + 1e-8).mean()  # log loss for confidence

    return loss, conf_loss
