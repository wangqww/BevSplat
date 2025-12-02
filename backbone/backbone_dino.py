from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from .backbone import Backbone
from .backbone_resnet import BackboneResnet, BackboneResnetCfg


@dataclass
class BackboneDinoCfg:
    name: Literal["dino"]
    model: Literal["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"]
    d_out: int


class BackboneDino(Backbone[BackboneDinoCfg]):
    def __init__(self, d_in: int = 3) -> None:
        super().__init__()
        assert d_in == 3
        self.dino = torch.hub.load("facebookresearch/dino:main", 'dino_vitb8', trust_repo=True)
        self.resnet_backbone = BackboneResnet(
            d_in,
        )
        self.global_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
        )
        self.local_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
        )

    def forward(
        self,
        img,
    ) -> Float[Tensor, "batch view d_out height width"]:
        # Compute features from the DINO-pretrained resnet50.
        resnet_features = self.resnet_backbone(img)

        # Compute features from the DINO-pretrained ViT.
        b, v, _, h, w = img.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        tokens = rearrange(img, "b v c h w -> (b v) c h w")
        tokens = self.dino.get_intermediate_layers(tokens)[0]
        global_token = self.global_token_mlp(tokens[:, 0])
        local_tokens = self.local_token_mlp(tokens[:, 1:])

        # Repeat the global token to match the image shape.
        global_token = repeat(global_token, "(b v) c -> b v c h w", b=b, v=v, h=h, w=w)

        # Repeat the local tokens to match the image shape.
        local_tokens = repeat(
            local_tokens,
            "(b v) (h w) c -> b v c (h hps) (w wps)",
            b=b,
            v=v,
            h=h // self.patch_size,
            hps=self.patch_size,
            w=w // self.patch_size,
            wps=self.patch_size,
        )

        return resnet_features + local_tokens + global_token

    @property
    def patch_size(self) -> int:
        return int(8)

    @property
    def d_out(self) -> int:
        return 512
