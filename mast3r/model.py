# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import ast
import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa


inf = float("inf")


def _safe_eval_model_arg(value):
    allowed_classes = {
        "AsymmetricMASt3R": AsymmetricMASt3R,
        "AsymmetricCroCo3DStereo": AsymmetricCroCo3DStereo,
    }
    allowed_names = {"inf": inf}

    def convert(node):
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_classes:
                raise ValueError(
                    "checkpoint model args must instantiate AsymmetricMASt3R "
                    "or AsymmetricCroCo3DStereo"
                )
            if node.args:
                raise ValueError("checkpoint model args must use keyword arguments only")
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise ValueError("checkpoint model args must not use **kwargs")
                kwargs[keyword.arg] = convert(keyword.value)
            return allowed_classes[node.func.id](**kwargs)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(convert(item) for item in node.elts)
        if isinstance(node, ast.List):
            return [convert(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {convert(k): convert(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name) and node.id in allowed_names:
            return allowed_names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            item = convert(node.operand)
            if not isinstance(item, (int, float)):
                raise ValueError("checkpoint model args contain invalid unary expression")
            return -item if isinstance(node.op, ast.USub) else item
        raise ValueError(f"checkpoint model args contain unsupported expression: {type(node).__name__}")

    return convert(ast.parse(value, mode="eval"))


def _load_checkpoint(model_path, allow_unsafe_checkpoint_load=False):
    try:
        return torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if allow_unsafe_checkpoint_load:
            return torch.load(model_path, map_location="cpu")
        raise RuntimeError(
            "Safe torch.load(weights_only=True) is unavailable. Upgrade PyTorch or "
            "set allow_unsafe_checkpoint_load=True only for trusted checkpoints."
        ) from exc
    except Exception as exc:
        if allow_unsafe_checkpoint_load:
            return torch.load(model_path, map_location="cpu", weights_only=False)
        raise RuntimeError(
            "Safe checkpoint loading failed. If this is a trusted legacy pickle "
            "checkpoint, set allow_unsafe_checkpoint_load=True."
        ) from exc


def load_model(model_path, device, verbose=True, allow_unsafe_checkpoint_load=False):
    if verbose:
        print("... loading model from", model_path)
    ckpt = _load_checkpoint(model_path, allow_unsafe_checkpoint_load)
    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = _safe_eval_model_arg(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(
        self, desc_mode=("norm"), two_confs=False, desc_conf_mode=None, **kwargs
    ):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        allow_unsafe_checkpoint_load=False,
        **kw,
    ):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(
                pretrained_model_name_or_path,
                device="cpu",
                allow_unsafe_checkpoint_load=allow_unsafe_checkpoint_load,
            )
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(
                pretrained_model_name_or_path, **kw
            )

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        self.downstream_head2 = mast3r_head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        # magic wrapper
        self.head1 = transpose_to_landscape(
            self.downstream_head1, activate=landscape_only
        )
        self.head2 = transpose_to_landscape(
            self.downstream_head2, activate=landscape_only
        )
