import torch
import torch.nn as nn

from phystta.models.encoder_factory import EncoderWrapper


def test_encoder_frozen():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    enc = EncoderWrapper(model)
    assert all(not p.requires_grad for p in enc.model.parameters())


class _EncodeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4, bias=False)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return self.proj(flat)


def test_encoder_wrapper_encode_image_path():
    model = _EncodeImageModel()
    enc = EncoderWrapper(model)
    x = torch.randn(3, 1, 2, 4)
    z = enc.encode(x)
    assert z.shape == (3, 4)
