# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn as nn
from torchvision.models.resnet import Bottleneck
from types import SimpleNamespace

from .utils import checkpointed

class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)

class BEVNet(nn.Module):
    default_conf = {
        "pretrained": True,
        "num_blocks": "???",
        "latent_dim": "???",
        "input_dim": "${.latent_dim}",
        "output_dim": "${.latent_dim}",
        "confidence": False,
        "norm_layer": "nn.BatchNorm2d",  # normalization ind decoder blocks
        "checkpointed": False,  # whether to use gradient checkpointing
        "padding": "zeros",
    }

    def __init__(self):
        super(BEVNet, self).__init__()
        conf = SimpleNamespace(**{'name': None, 
                                  'trainable': True, 
                                  'freeze_batch_normalization': False, 
                                  'pretrained': True, 
                                  'num_blocks': 4, 
                                  'latent_dim': 64, 
                                  'input_dim': 64, 
                                  'output_dim': 64, 
                                  'confidence': True, 
                                  'norm_layer': 'nn.BatchNorm2d', 
                                  'checkpointed': False, 
                                  'padding': 'zeros'})
        blocks = []
        Block = checkpointed(Bottleneck, do=conf.checkpointed)
        for i in range(conf.num_blocks):
            dim = conf.input_dim if i == 0 else conf.latent_dim
            blocks.append(
                Block(
                    dim,
                    conf.latent_dim // Bottleneck.expansion,
                    norm_layer=eval(conf.norm_layer),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.output_layer = AdaptationBlock(conf.latent_dim, conf.output_dim)
        self.confidence_layer = AdaptationBlock(conf.latent_dim, 1)

        def update_padding(module):
            if isinstance(module, nn.Conv2d):
                module.padding_mode = conf.padding

        if conf.padding != "zeros":
            self.bocks.apply(update_padding)

    def forward(self, data):
        features = self.blocks(data)
        pred = {
            "output": self.output_layer(features),
        }
        pred["confidence"] = self.confidence_layer(features).sigmoid()
        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
