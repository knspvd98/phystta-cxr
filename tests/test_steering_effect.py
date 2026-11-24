import torch

from phystta.intervention.eraser import SteeringEraser


def test_steering_reduces_shift_variance():
    torch.manual_seed(0)
    N, D = 128, 8
    v = torch.randn(D)
    v = v / v.norm()
    z = torch.randn(N, D)
    z_shift = z + 0.8 * v

    # linear steering via MLP
    mlp = torch.nn.Linear(D, 1, bias=False)
    mlp.weight.data = v.view(1, -1)
    eraser = SteeringEraser(v.view(-1, 1), mlp)

    z_e = eraser(z)
    z_shift_e = eraser(z_shift)
    diff_pre = torch.norm(z_shift - z, dim=1).mean().item()
    diff_post = torch.norm(z_shift_e - z_e, dim=1).mean().item()
    assert diff_post < 0.2 * diff_pre
