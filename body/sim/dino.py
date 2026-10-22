"""THE PRETRAINED EYE (the owner's yes, 2026-10-20: 'a frozen pretrained visual encoder behind the eyes'; Fable's review: the life's
vision is learned from one room of flat-coloured primitives, the representational breadth no life can earn). DINOv2-small (Meta's public
release, facebook/dinov2-small: 22 M parameters, self-supervised on 142 M images; Oquab et al. 2023), frozen, reads the colour camera
each tick: its CLS code for the whole image (what the scene holds) and for the colour window at the gaze (what the eyes rest on),
384 + 384 numbers, the channel eye_d (body/sim/anatomy.py, cfg eye_dino). It learns nothing: a born visual system, as a newborn's
visual cortex arrives organised before it has seen anything. The weights are read from a local folder (no network at run time).

    eye = DinoEye()                       # mps where the Mac has it, else cpu
    v = eye.features(colour_image, colour_window)    # -> float32 [768]
"""
import os

import numpy as np
import torch

DINO_PATH = os.environ.get("DINO_PATH", "/Users/lukehamond/Projects/project/data/models/dinov2-small")
EYE_D = 768                                   # the channel's width: two CLS codes of 384
FULL_HW = (126, 238)                          # the colour camera's 238 x 134 image at the patch grid nearest its own size (17 x 9 patches)
WIN_HW = (112, 112)                           # the 64 px colour window, 8 x 8 patches
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # the encoder's own preprocessing (its preprocessor_config.json)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoEye:
    """the frozen encoder and its two reads a tick"""

    def __init__(self, path=DINO_PATH, device=None):
        from transformers import Dinov2Model
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.m = Dinov2Model.from_pretrained(path).eval().to(self.dev)
        for p_ in self.m.parameters():
            p_.requires_grad_(False)
        self.mean = MEAN.to(self.dev); self.std = STD.to(self.dev)

    def _prep(self, img, hw):
        x = torch.from_numpy(np.ascontiguousarray(img[..., :3])).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = torch.nn.functional.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)
        return (x.to(self.dev) - self.mean) / self.std

    @torch.no_grad()
    def features(self, full_rgb, win_rgb):
        """the colour image's and the colour window's CLS codes (the encoder's layer-normed CLS) -> float32 [768]"""
        a = self.m(pixel_values=self._prep(full_rgb, FULL_HW)).pooler_output[0]
        b = self.m(pixel_values=self._prep(win_rgb, WIN_HW)).pooler_output[0]
        return torch.cat([a, b]).float().cpu().numpy()
