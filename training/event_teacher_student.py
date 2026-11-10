import copy

import torch
import torch.nn as nn
from torch import optim

from model.event_dark3r_student import EventDark3RStudent, make_view_pair_from_batch


def make_frozen_teacher(dark3r_model):
    teacher = copy.deepcopy(dark3r_model)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def build_event_student(dark3r_model, config=None):
    return EventDark3RStudent(dark3r_model, config=config)


def set_event_training_stage(student, stage=1, partial_finetune_blocks=4):
    """Configure stage-wise trainability for EventDark3RStudent.

    Stage 1 warms up only the event patch embedding and fusion adapter. Stage 2
    jointly trains the full event encoder with the Dark3R decoder/head. Stage 3
    additionally opens the last RGB encoder blocks with a small learning rate.
    Stage 0 trains the full event encoder and fusion adapter while keeping the
    RGB Dark3R/LoRA backbone frozen. Stage 4 trains the full event path and only
    the last RGB encoder blocks, keeping the decoder and geometry heads frozen.
    Stage 5 opens the complete RGB/cross-view trunk while keeping all downstream
    heads frozen. Stage 6 opens the complete student, including downstream heads.
    """
    if getattr(student, "pose_head_only", False):
        student.requires_grad_(False)
        pose_head = getattr(student, "pose_head", None)
        if not isinstance(pose_head, nn.Module):
            raise RuntimeError("training.pose_head_only requires pose_head.enabled=true")
        pose_head.requires_grad_(True)
        return

    if getattr(student, "descriptor_head_only", False):
        student.requires_grad_(False)
        descriptor_modules = []
        for module_name in ("downstream_head1", "downstream_head2"):
            module = getattr(student.dark3r, module_name, None)
            descriptor_head = getattr(module, "head_local_features", None)
            if isinstance(descriptor_head, nn.Module):
                descriptor_head.requires_grad_(True)
                descriptor_modules.append(descriptor_head)
        if len(descriptor_modules) != 2:
            raise RuntimeError(
                "training.descriptor_head_only requires both MASt3R "
                "downstream heads to expose head_local_features"
            )
        return

    if getattr(student, "geometry_head_only", False):
        student.requires_grad_(False)
        geometry_heads = []
        for module_name in ("downstream_head1", "downstream_head2"):
            module = getattr(student.dark3r, module_name, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(True)
                geometry_heads.append(module)
        if len(geometry_heads) != 2:
            raise RuntimeError(
                "training.geometry_head_only requires both MASt3R downstream heads"
            )
        return

    for param in student.event_encoder.parameters():
        param.requires_grad = False
    for param in student.fusion_adapter.parameters():
        param.requires_grad = True
    event_film = getattr(student, "event_film", None)
    if isinstance(event_film, nn.Module):
        for param in event_film.parameters():
            param.requires_grad = True
    patch_embed = getattr(student.event_encoder, "patch_embed", None)
    if patch_embed is not None:
        for param in patch_embed.parameters():
            param.requires_grad = True

    for param in student.dark3r.parameters():
        param.requires_grad = False

    if stage == 0:
        for param in student.event_encoder.parameters():
            param.requires_grad = True
        return

    if stage == 1:
        return

    for param in student.event_encoder.parameters():
        param.requires_grad = True

    if stage == 4:
        n_finetune_blocks = int(partial_finetune_blocks)
        if n_finetune_blocks > 0:
            enc_blocks = getattr(student.dark3r, "enc_blocks", None)
            if enc_blocks is not None:
                for block in list(enc_blocks)[-n_finetune_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
            for name, param in student.dark3r.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        return

    if stage in (5, 6):
        for name, param in student.dark3r.named_parameters():
            is_downstream_head = name.startswith(
                ("downstream_head1.", "downstream_head2.", "head1.", "head2.")
            )
            param.requires_grad = stage == 6 or not is_downstream_head
        return

    decoder_module_names = (
        "decoder_embed",
        "dec_blocks",
        "dec_blocks2",
        "dec_norm",
    )
    for module_name in decoder_module_names:
        module = getattr(student.dark3r, module_name, None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True

    for module_name in ("downstream_head1", "downstream_head2", "head1", "head2"):
        module = getattr(student.dark3r, module_name, None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True

    if stage == 2:
        return

    n_finetune_blocks = int(partial_finetune_blocks)
    if n_finetune_blocks <= 0:
        return

    enc_blocks = getattr(student.dark3r, "enc_blocks", None)
    if enc_blocks is not None:
        for block in list(enc_blocks)[-n_finetune_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    for name, param in student.dark3r.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def build_event_optimizer_param_groups(
    student,
    lr_event_encoder=1e-4,
    lr_fusion_adapter=1e-4,
    lr_dark3r=1e-5,
    lr_head=1e-4,
    lr_pose_head=1e-4,
):
    groups = []
    event_params = [p for p in student.event_encoder.parameters() if p.requires_grad]
    fusion_params = [p for p in student.fusion_adapter.parameters() if p.requires_grad]
    event_film = getattr(student, "event_film", None)
    if isinstance(event_film, nn.Module):
        fusion_params.extend(p for p in event_film.parameters() if p.requires_grad)
    if event_params:
        groups.append({"name": "event_encoder", "params": event_params, "lr": lr_event_encoder})
    if fusion_params:
        groups.append({"name": "fusion_adapter", "params": fusion_params, "lr": lr_fusion_adapter})

    head_params = []
    dark_params = []
    for name, param in student.dark3r.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "downstream_head" in name:
            head_params.append(param)
        else:
            dark_params.append(param)
    if dark_params:
        groups.append({"name": "dark3r_partial", "params": dark_params, "lr": lr_dark3r})
    if head_params:
        groups.append({"name": "dark3r_head", "params": head_params, "lr": lr_head})
    pose_head = getattr(student, "pose_head", None)
    if isinstance(pose_head, nn.Module):
        pose_params = [param for param in pose_head.parameters() if param.requires_grad]
        if pose_params:
            groups.append({"name": "pose_head", "params": pose_params, "lr": lr_pose_head})
    return groups


def build_event_optimizer(student, **kwargs):
    return optim.Adam(build_event_optimizer_param_groups(student, **kwargs))


def teacher_student_forward(teacher, student, batch):
    """Run frozen RGB-only teacher and event-capable student."""
    with torch.no_grad():
        teacher_view1, teacher_view2 = make_view_pair_from_batch(batch, prefer_dark=False)
        teacher_outputs = teacher(teacher_view1, teacher_view2)
    student_outputs = student(batch)
    return teacher_outputs, student_outputs
