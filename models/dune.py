import torch
import einops as E
import torch.nn.functional as F
import timm
from timm import create_model
import numpy as np
import types

def center_padding(images, patch_size):
    """
    修正后的中心填充函数。
    如果一个维度已经是 patch_size 的整数倍，则不对该维度进行填充。
    """
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    # 分别计算每个维度的填充量
    # 如果余数为0，则该维度不需要填充；否则，计算需要补充的像素数
    pad_h = 0 if diff_h == 0 else patch_size - diff_h
    pad_w = 0 if diff_w == 0 else patch_size - diff_w

    # 如果两个维度都不需要填充，直接返回
    if pad_h == 0 and pad_w == 0:
        return images

    # 计算上下左右需要填充的具体像素数，以实现中心填充
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l

    # 使用 F.pad 进行填充
    # 注意 F.pad 的填充顺序是 (左, 右, 上, 下)，对应 (w, w, h, h)
    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    
    return images

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        B, C, H, W = x.shape
        grid_size = (
            (H - self.patch_embed.patch_size[0])
            // self.patch_embed.proj.stride[0]
            + 1,
            (W - self.patch_embed.patch_size[1])
            // self.patch_embed.proj.stride[1]
            + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)

def process_image(image, stride):

    h, w = image.shape[2:]

    height_int = (h // stride)*stride
    width_int = (w // stride)*stride

    image_resized = torch.nn.functional.interpolate(image, size=(height_int, width_int), mode='bilinear')

    return image_resized

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dinov1",
        model_name="vitb14",
        output="dense-cls",
        layer=-1,
        return_multilayer=True,
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        
        dino_vit = torch.hub.load("/home/qiwei/.cache/torch/hub/naver_dune_main", "dune_vitbase_14_448_paper_encoder", source='local')

        self.vit = dino_vit.eval().to(torch.float32)
        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks[0])
        # TODO: change this to be 8,9,10,11
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        
        # images = process_image(images, self.patch_size)
        images = center_padding(images, self.patch_size)
        # pad images (if needed) to ensure it matches patch_size
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        
        x = self.vit.patch_embed(images)
        x = self.vit._pos_embed(x)
        x = self.vit.pos_drop(x)
        x = self.vit.norm_pre(x)

        embeds = []
        for i, blk in enumerate(self.vit.blocks[0]):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(self.vit.norm(x))
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            fit_cls_tok = x_i[:, 0]
            # ignoring register tokens
            fit_spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, fit_spatial, fit_cls_tok, (h, w))
            
            if x_i.shape[2] == 37:
                x_i = F.interpolate(x_i, size=(32, 32), mode="bilinear", align_corners=True)
            if x_i.shape[2] == 19:
                x_i = F.interpolate(x_i, size=(16, 64), mode="bilinear", align_corners=True)
            if x_i.shape[2] == 12:
                x_i = F.interpolate(x_i, size=(10, 10), mode="bilinear", align_corners=True)
            if x_i.shape[2] == 23:
                x_i = F.interpolate(x_i, size=(20, 40), mode="bilinear", align_corners=True)            

            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs