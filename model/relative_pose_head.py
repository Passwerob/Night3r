import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_6d_to_matrix(rotation_6d, eps=1.0e-6):
    """Convert two predicted rotation columns to a proper SO(3) matrix."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"rotation_6d must end in 6 values, got {rotation_6d.shape}")
    column1, column2 = rotation_6d.split(3, dim=-1)
    basis1 = F.normalize(column1, dim=-1, eps=eps)
    column2 = column2 - (basis1 * column2).sum(dim=-1, keepdim=True) * basis1
    basis2 = F.normalize(column2, dim=-1, eps=eps)
    basis3 = torch.cross(basis1, basis2, dim=-1)
    return torch.stack((basis1, basis2, basis3), dim=-1)


class TokenAttentionPool(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, tokens):
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,C], got {tokens.shape}")
        normalized = self.norm(tokens)
        weights = torch.softmax(self.score(normalized), dim=1)
        return (weights * normalized).sum(dim=1)


class RelativePoseHead(nn.Module):
    """Regress T_camera2_from_camera1 from the final cross-view tokens."""

    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        pool_hidden_dim=128,
        dropout=0.1,
        separate_branches=False,
    ):
        super().__init__()
        self.separate_branches = bool(separate_branches)
        if self.separate_branches:
            self.rotation_pool1 = TokenAttentionPool(input_dim, pool_hidden_dim)
            self.rotation_pool2 = TokenAttentionPool(input_dim, pool_hidden_dim)
            self.translation_pool1 = TokenAttentionPool(input_dim, pool_hidden_dim)
            self.translation_pool2 = TokenAttentionPool(input_dim, pool_hidden_dim)
            self.rotation_regressor = self._make_regressor(
                input_dim, hidden_dim, dropout, output_dim=6
            )
            self.translation_regressor = self._make_regressor(
                input_dim, hidden_dim, dropout, output_dim=3
            )
            self._initialize_separate_outputs()
        else:
            self.pool1 = TokenAttentionPool(input_dim, pool_hidden_dim)
            self.pool2 = TokenAttentionPool(input_dim, pool_hidden_dim)
            self.regressor = self._make_regressor(
                input_dim, hidden_dim, dropout, output_dim=9
            )
            self._initialize_shared_output()

    @staticmethod
    def _make_regressor(input_dim, hidden_dim, dropout, output_dim):
        return nn.Sequential(
            nn.LayerNorm(4 * input_dim),
            nn.Linear(4 * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _initialize_shared_output(self):
        final = self.regressor[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1.0e-4)
        with torch.no_grad():
            final.bias.zero_()
            final.bias[:6].copy_(final.bias.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def _initialize_separate_outputs(self):
        rotation_final = self.rotation_regressor[-1]
        translation_final = self.translation_regressor[-1]
        nn.init.normal_(rotation_final.weight, mean=0.0, std=1.0e-4)
        nn.init.normal_(translation_final.weight, mean=0.0, std=1.0e-4)
        with torch.no_grad():
            rotation_final.bias.copy_(
                rotation_final.bias.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )
            translation_final.bias.zero_()

    @staticmethod
    def _pair_feature(pool1, pool2, tokens1, tokens2):
        pooled1 = pool1(tokens1)
        pooled2 = pool2(tokens2)
        return torch.cat(
            (pooled1, pooled2, pooled2 - pooled1, pooled1 * pooled2), dim=-1
        )

    def forward(self, tokens1, tokens2):
        if self.separate_branches:
            rotation_feature = self._pair_feature(
                self.rotation_pool1, self.rotation_pool2, tokens1, tokens2
            )
            translation_feature = self._pair_feature(
                self.translation_pool1, self.translation_pool2, tokens1, tokens2
            )
            rotation_6d = self.rotation_regressor(rotation_feature)
            translation = self.translation_regressor(translation_feature)
        else:
            pair_feature = self._pair_feature(
                self.pool1, self.pool2, tokens1, tokens2
            )
            pose_vector = self.regressor(pair_feature)
            rotation_6d = pose_vector[..., :6]
            translation = pose_vector[..., 6:]
        rotation = rotation_6d_to_matrix(rotation_6d)

        batch_size = tokens1.shape[0]
        relative_pose = torch.eye(
            4, device=tokens1.device, dtype=tokens1.dtype
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        relative_pose[:, :3, :3] = rotation
        relative_pose[:, :3, 3] = translation
        return relative_pose
