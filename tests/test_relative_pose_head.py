import torch

from model.relative_pose_head import RelativePoseHead, rotation_6d_to_matrix


def test_rotation_6d_identity_and_so3():
    rotation = rotation_6d_to_matrix(
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    )
    identity = torch.eye(3).unsqueeze(0)
    torch.testing.assert_close(rotation, identity)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity)
    torch.testing.assert_close(torch.det(rotation), torch.ones(1))


def test_relative_pose_head_shape_and_gradient():
    head = RelativePoseHead(input_dim=32, hidden_dim=48, pool_hidden_dim=16, dropout=0.0)
    tokens1 = torch.randn(2, 12, 32)
    tokens2 = torch.randn(2, 12, 32)
    pose = head(tokens1, tokens2)
    assert pose.shape == (2, 4, 4)
    torch.testing.assert_close(pose[:, 3], pose.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(2, -1))
    loss = pose[:, :3].square().mean()
    loss.backward()
    assert all(param.grad is not None for param in head.parameters())


def test_separate_pose_branches_have_independent_gradients():
    head = RelativePoseHead(
        input_dim=32,
        hidden_dim=48,
        pool_hidden_dim=16,
        dropout=0.0,
        separate_branches=True,
    )
    tokens1 = torch.randn(2, 12, 32)
    tokens2 = torch.randn(2, 12, 32)

    pose = head(tokens1, tokens2)
    pose[:, :3, :3].square().mean().backward()
    rotation_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name.startswith("rotation_")
    ]
    translation_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name.startswith("translation_")
    ]
    assert rotation_parameters
    assert translation_parameters
    assert all(parameter.grad is not None for parameter in rotation_parameters)
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in translation_parameters
    )

    head.zero_grad(set_to_none=True)
    pose = head(tokens1, tokens2)
    pose[:, :3, 3].square().mean().backward()
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in rotation_parameters
    )
    assert all(parameter.grad is not None for parameter in translation_parameters)
