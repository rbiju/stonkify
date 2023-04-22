import torch
from stonkify.cv import PatchEmbed


def test_patch_embed():
    patch_embed = PatchEmbed(h=128, w=200, step=20, embed_dim=64)
    test_tensor = torch.rand(5, 1, 128, 200)
    out = patch_embed(test_tensor)

    return out


if __name__ == "__main__":
    test_patch_embed()
