import torch

from phystta.intervention.eraser import LinearSubspaceEraser, SteeringEraser
from phystta.intervention.online import MixtureEraserAdaptor, FastSteeringAdaptor, EntropySteeringAdaptor


def test_mixture_adaptor_shape():
    D, k = 6, 2
    e1 = LinearSubspaceEraser(torch.randn(D, k))
    e2 = LinearSubspaceEraser(torch.randn(D, k))
    weight_net = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    adaptor = MixtureEraserAdaptor([e1, e2], weight_net)
    z = torch.randn(5, D)
    scores = torch.randn(5, 1)
    out = adaptor(z, scores)
    assert out.shape == z.shape


def test_fast_steering_adaptor_shape():
    D, k = 6, 3
    V = torch.randn(D, k)
    mlp = torch.nn.Sequential(torch.nn.Linear(D, 4), torch.nn.ReLU(), torch.nn.Linear(4, k), torch.nn.Tanh())
    eraser = SteeringEraser(V, mlp)
    probe_w = torch.randn(D)
    adaptor = FastSteeringAdaptor(eraser, probe_w, head_w=None)
    z = torch.randn(7, D)
    out = adaptor(z)
    assert out.shape == z.shape


def test_entropy_steering_adaptor_shape():
    D, k = 6, 2
    V = torch.randn(D, k)
    mlp = torch.nn.Sequential(torch.nn.Linear(D, 4), torch.nn.ReLU(), torch.nn.Linear(4, k), torch.nn.Tanh())
    eraser = SteeringEraser(V, mlp)
    head = torch.nn.Linear(D, 3)
    adaptor = EntropySteeringAdaptor(eraser, head, steps=2, lr=1e-2)
    z = torch.randn(4, D)
    out = adaptor(z)
    assert out.shape == z.shape
