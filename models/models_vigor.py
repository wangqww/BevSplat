import torch.nn as nn
import torch
import os
# import plotly.graph_objects as go
from models.VGGW import VGGUnet, L2_norm, Encoder, Decoder
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Tuple, Union
from torchvision.transforms.functional import resize

from models.pano_utils import split_panorama, showDepth, tensor_to_cv2_image
from jaxtyping import Float
from torch import Tensor
import numpy as np

from models.dino_fit import DINO
from models.dpt_single import DPT

from gaussian.encoder_pano import GaussianEncoder
# from gaussian.encoder import GaussianEncoder
from vis_gaussian_pano import render_projections
from jacobian import grid_sample
from visualize import *
import cv2
from pathlib import Path
from ply_export import export_ply
# from line_profiler import LineProfiler

to_pil_image = transforms.ToPILImage()
# original_raw_Lpips_step = 50000
raw_Lpips_step = 25000
# original_L1_step = 100000
L1_step = 40000
# original_refine_Lpips_step = 100000
refine_Lpips_step = 40000
# original_discriminator_loss_active_step = 125000
discriminator_loss_active_step = 60000

def equirectangular_to_xyz(width, height, device):
    """Convert equirectangular coordinates to spherical 3D coordinates in OpenCV convention and rotate 90° around Y-axis using PyTorch."""
    # 创建 theta 和 phi 为 1D 张量
    theta = torch.linspace(0, 2 * torch.pi, width, device=device)  # 方位角 [0, 2π]
    phi = torch.linspace(0, torch.pi, height, device=device)       # 仰角 [0, π]
    
    # 生成网格，调整 indexing='ij' 确保符合 PyTorch 约定
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')

    # 计算 OpenCV 形式的 X, Y, Z 坐标
    x = -torch.sin(phi) * torch.cos(theta)   # OpenCV X: 右
    y = -torch.cos(phi)                     # OpenCV Y: 下
    z = -torch.sin(phi) * torch.sin(theta)  # OpenCV Z: 前

    # 将 x, y, z 堆叠在一起，并调整维度 (height, width, 3)
    xyz = torch.stack((x, y, z), dim=-1)  # (B, V, H, W, 3)

    return xyz

def onlyDepth(depth, save_name):
    cmap = cm.Spectral
    depth = depth[0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)
    
    c_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_name, c_depth)

def get_integer(f: Fraction) -> int:
    assert f.denominator == 1, "Fraction is not integer"
    return f.numerator

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class ModelVIGOR(nn.Module):
    def __init__(self, args, device):  # device='cuda:0',
        super(ModelVIGOR, self).__init__()
        self.device = device
        self.args = args
        self.share = args.share
        self.grd_res = args.grd_res
        self.level = sorted([int(item) for item in args.level.split('_')])[0]
        self.N_iters = args.N_iters
        self.channels = ([int(item) for item in self.args.channels.split('_')])

        self.gaussian_encoder = GaussianEncoder(area=args.area)
        self.dino_feat = DINO()
        self.dpt_sat = DPT(self.dino_feat.feat_dim)
        self.dpt_grd = DPT(self.dino_feat.feat_dim)
        # self.near = torch.ones(args.batch_size, self.face_num).to(device) * 0.5
        # self.far = torch.ones(args.batch_size, self.face_num).to(device) * 160
        self.FeatureForT = VGGUnet(self.level, self.channels)
        self.SatFeatureNet = VGGUnet(self.level, self.channels)
        self.GrdFeatureNet = VGGUnet(self.level, self.channels)
        self.global_step = 0
        self.camera_k = torch.tensor([
            [0.5000, 0.0000, 0.5000],
            [0.0000, 0.5000, 0.5000],
            [0.0000, 0.0000, 1.0000]], device=device).unsqueeze(0)

        torch.autograd.set_detect_anomaly(True)


    def forward_project(self, image_tensor, depth, meter_per_pixel, sat_width=128):
        B, C, grd_H, grd_W = image_tensor.shape
        image_tensor = image_tensor.permute(0,2,3,1).contiguous().view(B*grd_H*grd_W, -1)
        depth = depth.permute(0,2,3,1)
        # xyz_grd = xyz_w * depth / meter_per_pixel
        
        xyz_coords = equirectangular_to_xyz(grd_W, grd_H, image_tensor.device)
        xyz_grd = xyz_coords.unsqueeze(0) * depth
        # xyz_grd = xyz_grd.long()
        # xyz_grd[:,:,:,0:1] += sat_width // 2
        # xyz_grd[:,:,:,2:3] += sat_width // 2
        # B, H, W, C = xyz_grd.shape
        xyz_grd[..., 0] = xyz_grd[..., 0] / meter_per_pixel[:,None,None]
        xyz_grd[..., 2] = xyz_grd[..., 2] / meter_per_pixel[:,None,None]
        xyz_grd[..., 0] = xyz_grd[..., 0].long()
        xyz_grd[..., 2] = xyz_grd[..., 2].long()
        xyz_grd = xyz_grd.view(B*grd_H*grd_W, -1)

        batch_ix = torch.cat([torch.full([grd_H*grd_W, 1], ix, device=image_tensor.device) for ix in range(B)], dim=0)
        xyz_grd = torch.cat([xyz_grd, batch_ix], dim=-1)

        kept = (xyz_grd[:,0] >= -(sat_width // 2)) & (xyz_grd[:,0] <= (sat_width // 2) - 1) & (xyz_grd[:,2] >= -(sat_width // 2)) & (xyz_grd[:,2] <= (sat_width // 2) - 1)

        xyz_grd_kept = xyz_grd[kept]
        image_tensor_kept = image_tensor[kept]

        max_height = xyz_grd_kept[:,1].max()

        xyz_grd_kept[:,0] = xyz_grd_kept[:,0] + sat_width // 2
        xyz_grd_kept[:,1] = max_height - xyz_grd_kept[:,1]
        xyz_grd_kept[:,2] = xyz_grd_kept[:,2] + sat_width // 2
        xyz_grd_kept = xyz_grd_kept[:,[2,0,1,3]]
        rank = torch.stack((xyz_grd_kept[:, 0] * sat_width * B + (xyz_grd_kept[:, 1] + 1) * B + xyz_grd_kept[:, 3], xyz_grd_kept[:, 2]), dim=1)
        sorts_second = torch.argsort(rank[:, 1])
        xyz_grd_kept = xyz_grd_kept[sorts_second]
        image_tensor_kept = image_tensor_kept[sorts_second]
        sorted_rank = rank[sorts_second]
        sorts_first = torch.argsort(sorted_rank[:, 0], stable=True)
        xyz_grd_kept = xyz_grd_kept[sorts_first]
        image_tensor_kept = image_tensor_kept[sorts_first]
        sorted_rank = sorted_rank[sorts_first]
        kept = torch.ones_like(sorted_rank[:, 0])
        kept[:-1] = sorted_rank[:, 0][:-1] != sorted_rank[:, 0][1:]
        res_xyz = xyz_grd_kept[kept.bool()]
        res_image = image_tensor_kept[kept.bool()]
        
        # grd_image_index = torch.cat((-res_xyz[:,1:2] + grd_image_width - 1,-res_xyz[:,0:1] + grd_image_height - 1), dim=-1)
        final = torch.zeros(B,sat_width,sat_width,C).to(torch.float32).to('cuda')
        sat_height = torch.zeros(B,sat_width,sat_width,1).to(torch.float32).to('cuda')
        final[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_image

        res_xyz[:,2][res_xyz[:,2] < 1e-1] = 1e-1
        sat_height[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_xyz[:,2].unsqueeze(-1)
        sat_height = sat_height.permute(0,3,1,2)
        # img_num = 0
        # project_grd_img = to_pil_image(final[img_num].permute(2, 0, 1))
        # project_grd_img.save('sat_feat.png')

        # project_grd_img = to_pil_image(origin_image_tensor[img_num])
        # project_grd_img.save('grd_feat.png')

        return final.permute(0,3,1,2)

    def sat2grd_uv(self, rot, shift_u, shift_v, level, H, W, meter_per_pixel):
        '''
        rot.shape = [B]
        shift_u.shape = [B]
        shift_v.shape = [B]
        H: scalar  height of grd feature map, from which projection is conducted
        W: scalar  width of grd feature map, from which projection is conducted
        '''

        B = shift_u.shape[0]

        # shift_u = shift_u / np.power(2, 3 - level)
        # shift_v = shift_v / np.power(2, 3 - level)

        S = 512 / np.power(2, 3 - level)
        shift_u = shift_u * S / 4
        shift_v = shift_v * S / 4

        # shift_u = shift_u / 512 * S
        # shift_v = shift_v / 512 * S

        ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=shift_u.device),
                                torch.arange(0, S, dtype=torch.float32, device=shift_u.device))
        ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
        jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

        radius = torch.sqrt((ii-(S/2-0.5 + shift_v.reshape(-1, 1, 1)))**2 + (jj-(S/2-0.5 + shift_u.reshape(-1, 1, 1)))**2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)), jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = (theta + rot[:, None, None] * self.args.rotation_range / 180 * np.pi) % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * np.power(2, 3-level)
        phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(-2))
        phimin = phimin / np.pi * H

        uv = torch.stack([theta, phimin], dim=-1)

        return uv

    def project_grd_to_map(self, grd_f, grd_c, rot, shift_u, shift_v, level, meter_per_pixel):
        '''
        grd_f.shape = [B, C, H, W]
        shift_u.shape = [B]
        shift_v.shape = [B]
        '''
        B, C, H, W = grd_f.size()
        uv = self.sat2grd_uv(rot, shift_u, shift_v, level, H, W, meter_per_pixel)  # [B, S, S, 2]
        grd_f_trans, _ = grid_sample(grd_f, uv)
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
        else:
            grd_c_trans = None
        return grd_f_trans, grd_c_trans, uv

    # @profile
    def forward2DoF(self, sat, grd, depth_imgs, grd_ori, gt_rot, meter_per_pixel):
        b = sat.shape[0]
        grd_res = 80
        self.gaussian_encoder.eval()

        self.near = torch.ones(b).to(self.device) * 0.2
        self.far = torch.ones(b).to(self.device) * 160
        self.sat = F.interpolate(sat, (128, 128), mode='bilinear', align_corners=True)
        # mask = (grd != 0).any(dim=1, keepdim=True).float() 
        # depth_imgs = depth_imgs * mask * 35
        # sat_feat_dict, sat_conf_dict = self.SatFeatureNet(sat)
        # grd_feat_dict, grd_conf_dict = self.GrdFeatureNet(grd)

        # dptstart
        with torch.no_grad():
            # dino
            sat_feat = self.dino_feat(sat)
            grd_feat = self.dino_feat(grd)
            if isinstance(sat_feat, (tuple, list)):
                sat_feats = [_f.detach() for _f in sat_feat]
            if isinstance(grd_feat, (tuple, list)):
                grd_feats = [_f.detach() for _f in grd_feat]
        
        # TODO: use two dpt and upsample grd_feat
        # dpt
        if self.share:
            sat_feat, sat_conf = self.dpt_sat(sat_feats)
            grd_feat, grd_conf = self.dpt_sat(grd_feats)
        else:
            sat_feat, sat_conf = self.dpt_sat(sat_feats)
            grd_feat, grd_conf = self.dpt_grd(grd_feats)
        sat_feat_dict = {}
        sat_conf_dict = {}
        grd_feat_dict = {}
        grd_conf_dict = {}

        sat_feat_dict[self.level] = sat_feat
        sat_conf_dict[self.level] = sat_conf
        grd_feat_dict[self.level] = grd_feat
        grd_conf_dict[self.level] = grd_conf
        # dpt over

        # sat_feat_dict, sat_conf_dict = self.FeatureForT(sat)
        # grd_feat_dict, grd_conf_dict = self.FeatureForT(pers_imgs.view(b*v, c, h, w))
        shift_u = torch.zeros([b], dtype=torch.float32, requires_grad=True, device=sat.device)
        shift_v = torch.zeros([b], dtype=torch.float32, requires_grad=True, device=sat.device)
        g2s_feat_dict = {}
        g2s_conf_dict = {}
        
        gs_grd = F.interpolate(grd, (grd_res, grd_res*2), mode='bilinear', align_corners=True)
        gs_depth_img = F.interpolate(depth_imgs, (grd_res, grd_res*2), mode='bilinear', align_corners=True)  # 50
        
        grd_gaussian = self.gaussian_encoder(
            gs_grd, 
            grd_feat_dict[self.level], 
            grd_conf_dict[self.level], 
            gs_depth_img,
        )
        if self.args.rotation_range == 0:
            heading = torch.ones_like(gt_rot.unsqueeze(-1), device=gt_rot.device) * 90
            rot_range = 1
        else:
            heading = torch.ones_like(gt_rot.unsqueeze(-1), device=gt_rot.device) * 90 / self.args.rotation_range
            heading = heading + gt_rot.unsqueeze(-1)
            rot_range = self.args.rotation_range
        grd2sat_gaussian_color2, grd2sat_gaussian_feat2, grd2sat_gaussian_conf2, grd2sat_gaussian_depth = render_projections(grd_gaussian, (128,128), heading=heading, rot_range=rot_range, width=70.0, height=70.0)
        grd2sat_feat2 = grd2sat_gaussian_feat2
        grd2sat_conf2 = grd2sat_gaussian_conf2

        # vis
        # idx = 0
        # grd_feat_proj, grd_conf_proj, grd_uv = self.project_grd_to_map(
        #     grd_feat, None, gt_rot, shift_u, shift_v, self.level, meter_per_pixel)
        # forward_map = self.forward_project(grd_feat, gs_depth_img, meter_per_pixel.to('cuda') * 5.5, 128)

        # grd_color = decoder_grd.color
        # test_img = to_pil_image(grd_color[idx].clip(min=0, max=1))
        # test_img.save(f'grd_vigor.png')
        # test_img = to_pil_image(grd_ori[0].clip(min=0, max=1))
        # test_img.save(f'seq/ori_grd_vigor_test2.png')
        # test_img = to_pil_image(grd_feat_proj[0].clip(min=0, max=1))
        # test_img.save(f'ori_vigor.png')

        test_img = to_pil_image(grd2sat_gaussian_color2[0].clip(min=0, max=1))
        test_img.save(f'g2s_vigor_test.png')
        test_img = to_pil_image(self.sat[0].clip(min=0, max=1))
        test_img.save(f'sat_vigor_test.png')
        # vis feat
        # grd_vis = F.interpolate(grd, (80, 160), mode='bilinear', align_corners=True)
        # grd_mask = (grd_vis != 0).any(dim=1, keepdim=True).float()
        # single_features_to_RGB_colormap(grd2sat_gaussian_feat2, idx=0, img_name='pano_g2s_feat.png', cmap_name='rainbow')
        # single_features_to_RGB_colormap(sat_feat, idx=0, img_name='pano_sat_feat.png', cmap_name='rainbow')
        # single_features_to_RGB_colormap(torch.flip(forward_map, dims=[2]), idx=0, img_name='forward_map.png', cmap_name='rainbow')
        # single_features_to_RGB_colormap(grd_feat_proj, idx=0, img_name='grd_feat_proj.png', cmap_name='rainbow')

        # single_features_to_RGB_colormap(grd_feat * grd_mask, idx=0, img_name='pca_vis_cmap_viridis.png', cmap_name='PuBuGn')        # sat_features_to_RGB_2D_PCA(sat_feat_dict[self.level], grd2sat_feat2, idx)
        # single_features_to_RGB_colormap(grd2sat_gaussian_conf2, idx=0, img_name='pca_vis_cmap_viridis.png', cmap_name='PuBu')
        # single_features_to_RGB_colormap(grd_conf * grd_mask, idx=0, img_name='pca_vis_cmap_viridis.png', cmap_name='PuBu')
        # grd_features_to_RGB_2D_PCA_concat(pers_feat)
        # visualize_two_features_unified_colormap(
        #     grd2sat_gaussian_feat2[:,:,25:103,25:103],
        #     sat_feat[:,:,25:103,25:103],
        #     idx=0,
        #     img_name_base='seq/sat_weak_feat',
        #     cmap_name='rainbow',
        #     pc_low_percentile=10,
        #     pc_high_percentile=90
        # )

        sat_feat = sat_feat_dict[self.level]
        A = sat_feat.shape[-1]
        crop_H = int(A * 0.4)
        crop_W = int(A * 0.4)
        g2s_feat = TF.center_crop(grd2sat_feat2, [crop_H, crop_W])
        # mask = (g2s_feat != 0).any(dim=1, keepdim=True).float()

        g2s_conf = TF.center_crop(grd2sat_conf2, [crop_H, crop_W])

        g2s_feat_dict[self.level] = g2s_feat
        g2s_conf_dict[self.level] = g2s_conf
        
        sat_uncer_dict = {}
        for level in range(3):
            sat_uncer_dict[level] = None
        return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict
    
    def forward(self, sat, grd, depth_imgs, grd_ori, meter_per_pixel, gt_rot=None, gt_shift_u=None, gt_shift_v=None, stage=None, loop=None, save_dir=None):
        return self.forward2DoF(sat, grd, depth_imgs, grd_ori, gt_rot, meter_per_pixel)
        

def batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, masks=None):
    '''
    compute corr_maps for training
    result corr_map has a shape of [M, N, H, W],
    M is the number of satellite images and N is the number of ground images
    '''

    levels = sorted([int(item) for item in args.level.split('_')])
    corr_maps = {}
    for _, level in enumerate(levels):
        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape


        if args.ConfGrd > 0:

            if args.ConfSat > 0:

                # numerator
                signal = (sat_feat * sat_conf.pow(2)).repeat(1, B, 1, 1)   # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_conf_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denominator_grd = []
                sat_conf_pow = sat_conf.pow(2)
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                for i in range(0, B):
                    denom_grd = torch.sum(F.conv2d(sat_conf_pow[i:i+1, :, :, :].repeat(1, C, 1, 1), g2s_feat_conf_pow), dim=1)
                    denominator_grd.append(denom_grd)
                denominator_grd = torch.sqrt(torch.stack(denominator_grd, dim=0))

                # corr = corr / denominator_sat / denominator_grd

            else:

                # numerator
                signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                # denominator
                denominator_sat = []
                sat_feat_pow = (sat_feat).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))  # [B (M), B (N), H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

                # corr = corr / denominator_sat / denominator_grd

        else:

            signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)

            # fixme: denominator
            # denominator_sat1 = []
            # mask_kernel = TF.center_crop(masks[level], [crop_H, crop_W]).float().unsqueeze(1).repeat(B, 1, 1, 1)
            # for i in range(0, B):
            #     denom_sat = torch.sum(F.conv2d(sat_feat.pow(2)[i, :, None, :, :], mask_kernel), dim=0)
            #     denominator_sat1.append(denom_sat)
            # denominator_sat1 = torch.sqrt(torch.stack(denominator_sat1, dim=0))  # [B (M), B (N), H, W]
            
            mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
            l2_norm_kernel = mask.repeat(1, C, 1, 1)
            sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)
            denominator_sat = torch.sqrt(sat_feat_squared_sum + 1e-8)
            # single_features_to_RGB(g2s_feat)
            # single_features_to_RGB(g2s_feat * mask)
            # original
            # denominator_sat_ori = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            # denominator_sat_ori = torch.sqrt(torch.sum(denominator_sat_ori, dim=1, keepdim=True))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = 2 - 2 * corr / denominator  # [B, B, H, W]

        corr_maps[level] = corr

    return corr_maps


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, args, meters_per_pixel, GPS_error=5):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    gt_shift_u: [B]
    gt_shift_v: [B]
    meters_per_pixel: [B], corresponding to original image size
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = []

    # ------------------------------------------------

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        if (M == 1):
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10)))
        else:
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

        # ---------- preparing for GPS error Loss -------
        w = (torch.round(W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
        h = (torch.round(H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
        radius = (torch.ceil(GPS_error / (meters_per_pixel * np.power(2, 3 - level)))).long()
        GPS_dis = []
        for b_idx in range(M):
            # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
            start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius[b_idx])
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius[b_idx])
            start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius[b_idx])
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius[b_idx])
            GPS_dis.append(torch.min(
                corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))



def GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, args):
    '''
    Used when GT GPS lables are highly reliable.
    This function does not handle the rotation issue.
    '''
    levels = [int(item) for item in args.level.split('_')]

    losses = []
    # for level in range(len(corr_maps)):
    for _, level in enumerate(levels):
        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        w = torch.round(corr_W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4)
        h = torch.round(corr_H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4)

        # import pdb; pdb.set_trace()
        pos = corr[range(B), h.long(), w.long()]  # [B]
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.mean(torch.stack(losses, dim=0))



def corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, sat_uncer_dict=None):
    '''
    to be used during inference
    '''

    level = max([int(item) for item in args.level.split('_')])

    sat_feat = sat_feat_dict[level]
    sat_conf = sat_conf_dict[level]
    g2s_feat = g2s_feat_dict[level]
    g2s_conf = g2s_conf_dict[level]

    B, C, crop_H, crop_W = g2s_feat.shape
    A = sat_feat.shape[2]

    # s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
    # corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]
    #
    # if args.ConfGrd > 0:
    #     denominator = F.conv2d(sat_feat.pow(2).transpose(0, 1), g2s_conf.pow(2), groups=B).transpose(0, 1)
    # else:
    #     denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)

    if args.ConfGrd > 0:

        if args.ConfSat > 0:

            # numerator
            signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            # denominator
            sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

            sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
            denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]  # [B, H, W]
            denominator_grd = torch.sqrt(denominator_grd)

        else:

            # numerator
            signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            # denominator
            sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

            denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

            # corr = corr / denominator_sat / denominator_grd

    else:

        signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        kernel = g2s_feat
        corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

        denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        # denominator = corr / denominator_sat / denominator_grd

    denominator = denominator_sat * denominator_grd

    if args.use_uncertainty:
        denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

    denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

    corr = corr / denominator

    B, corr_H, corr_W = corr.shape

    max_index = torch.argmax(corr.reshape(B, -1), dim=1)
    pred_u = (max_index % corr_W - corr_W / 2)
    pred_v = (max_index // corr_W - corr_H / 2)

    # if level == 3:
    #     return pred_u, pred_v, corr
    #
    # elif level == 2:
    #     return pred_u * 2, pred_v * 2, corr

    return pred_u * np.power(2, 3 - level), pred_v * np.power(2, 3 - level), corr

def corr_for_accurate_translation_supervision(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args,
                                              sat_uncer_dict=None):
    levels = [int(item) for item in args.level.split('_')]

    corr_maps = {}
    for level in levels:

        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape
        A = sat_feat.shape[2]

        # s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        # corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]
        #
        # if args.ConfGrd > 0:
        #     denominator = F.conv2d(sat_feat.pow(2).transpose(0, 1), g2s_conf.pow(2), groups=B).transpose(0, 1)
        # else:
        #     denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)

        if args.ConfGrd > 0:

            if args.ConfSat > 0:

                # numerator
                signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]  # [B, H, W]
                denominator_grd = torch.sqrt(denominator_grd)

            else:

                # numerator
                signal = sat_feat.reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

                # denominator
                sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = g2s_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

                # corr = corr / denominator_sat / denominator_grd

        else:

            signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

            denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            denom_grd = torch.linalg.norm((g2s_feat).reshape(B, -1), dim=-1)  # [B]
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
            # denominator = corr / denominator_sat / denominator_grd

        denominator = denominator_sat * denominator_grd

        # if args.use_uncertainty:
        #     denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = corr / denominator

        corr_maps[level] = 2 - 2 * corr

    return corr_maps