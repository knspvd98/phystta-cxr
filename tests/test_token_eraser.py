import torch
import torch.nn as nn

from phystta.models.encoder_factory import EncoderWrapper
from phystta.intervention.eraser import LinearSubspaceEraser
from phystta.intervention.token_eraser import TokenEraserWrapper


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=16, patch=8):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ToyViT(nn.Module):
    def __init__(self, embed_dim=16, num_patches=4):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, patch=8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.norm = nn.LayerNorm(embed_dim)


def test_token_eraser_wrapper():
    model = ToyViT(embed_dim=16, num_patches=4)
    encoder = EncoderWrapper(model, input_adapter=None)
    V = torch.randn(16, 3)
    eraser = LinearSubspaceEraser(V)
    wrapper = TokenEraserWrapper(encoder, eraser, mode="cls", token_layers=1)
    x = torch.randn(2, 1, 16, 16)
    out = wrapper(x)
    assert out.shape == (2, 16)
    wrapper_mean = TokenEraserWrapper(encoder, eraser, mode="mean", token_layers=1)
    out2 = wrapper_mean(x)
    assert out2.shape == (2, 16)
