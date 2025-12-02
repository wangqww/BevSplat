import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import data_utils as utils
import os
import torchvision.transforms.functional as TF
from visualize import *

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
from models.VGGW import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec, FeatureHead
from jacobian import grid_sample

# from ConvLSTM import VE_LSTM3D, VE_LSTM2D, VE_conv, S_LSTM2D
# from models_ford import loss_func
import copy
import cv2

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

from models.dpt_single import DPT
from models.dino_fit import DINO

EPS = utils.EPS


class T2GA(nn.Module):
    """
    Top-to-Ground Aggregation (T2GA) 模块的PyTorch实现。
    
    该模块通过注意力机制，将地面视图中“高处”像素的特征，
    聚合到其垂直对应的“地面”像素上，以生成一个感知到遮挡的、
    更鲁棒的地面特征图。
    """
    def __init__(self, feature_dim: int, horizon_threshold=0.6):
        """
        初始化T2GA模块。

        参数:
        - feature_dim (int): 输入特征图的通道数 (C)。
        - horizon_threshold (int): 地平线阈值 (τ)。图像中行索引小于此值的像素被视为“高处像素”。
        """
        super().__init__()
        
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim 必须是一个正整数")

        self.feature_dim = feature_dim
        self.horizon_threshold = horizon_threshold
        
        # 论文中提到的非线性映射函数 M(.), 用于生成Query和Key
        # 这里我们用一个简单的MLP（一个线性层 + ReLU激活函数）来实现
        self.mapping_M = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_g: torch.Tensor) -> torch.Tensor:
        """
        执行T2GA的前向传播。

        参数:
        - F_g (torch.Tensor): 地面视图的特征图，形状为 [B, C, H, W]。

        返回:
        - torch.Tensor: 聚合后的新特征图 F_a，形状与输入 F_g 相同。
        """
        # 1. 获取输入尺寸并进行分割
        # F_g: [B, C, H, W]
        B, C, H, W = F_g.shape

        H_ground = int(self.horizon_threshold * H)
        H_elevated = H - H_ground

        # 将特征图垂直分割为“高处”部分和“地面”部分
        F_g_elevated = F_g[:, :, H_ground:, :]
        F_g_ground = F_g[:, :, :H_ground, :]

        # 2. 准备Q, K, V以进行批处理注意力计算
        # 注意力是逐列计算的，为了高效并行，我们将 B 和 W 合并为批次维度
        # permute + reshape: [B, C, H, W] -> [B, W, H, C] -> [B*W, H, C]
        q_input = F_g_ground.permute(0, 3, 2, 1).reshape(B * W, H_ground, C)
        kv_input = F_g_elevated.permute(0, 3, 2, 1).reshape(B * W, H_elevated, C)
        
        # 3. 计算 Q, K, V
        # Q 来自地面像素
        Q = self.mapping_M(q_input)  # Shape: [B*W, H_ground, C]
        
        # K, V 来自高处像素
        K = self.mapping_M(kv_input)  # Shape: [B*W, H_elevated, C]
        V = kv_input                   # Shape: [B*W, H_elevated, C]
        
        # 4. 执行标准的缩放点积注意力
        # (Q @ K^T) -> [B*W, H_ground, H_elevated]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.feature_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # (Attention @ V) -> [B*W, H_ground, C]
        F_att_flat = torch.bmm(attn_weights, V)
        
        # 5. 将结果重塑并与原始高处特征拼接
        # reshape + permute: [B*W, H_ground, C] -> [B, W, H_ground, C] -> [B, C, H_ground, W]
        F_att = F_att_flat.reshape(B, W, H_ground, C).permute(0, 3, 2, 1)
        
        # 根据论文公式(3)，将原始的高处特征与新聚合的地面特征在高度维度上拼接
        F_a = torch.cat([F_g_elevated, F_att], dim=2)
        
        return F_a


def extract_keypoints(confidence, topk=1024, start_ratio=0.6):
    """extrac key ponts from confidence map.
    Args:
        confidence: torch.Tensor with size (B,C,H,W).
        topk: extract topk points each confidence map
        start_ratio: extract close to ground part (start_ratio*H:)
    Returns:
        A torch.Tensor of index where the key points are.
    """
    assert (start_ratio < 1 and start_ratio >= 0)


    w_end = -1
    h_end = -1
    radius = 4
    if confidence.size(-1) > 1200: # KITTI
        radius = 5

        #  only extract close to ground part (start_H:)
    start_H = int(confidence.size(2)*start_ratio)
    confidence = confidence[:,:,start_H:h_end,:w_end].detach().clone()

    # fast Non-maximum suppression to remove nearby points
    def max_pool(x):
        return torch.nn.functional.max_pool2d(x, kernel_size=radius*2+1, stride=1, padding=radius)

    max_mask = (confidence == max_pool(confidence))
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_confidence = torch.where(supp_mask, torch.zeros_like(confidence), confidence)
        new_max_mask = (supp_confidence == max_pool(supp_confidence))
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    confidence = torch.where(max_mask, confidence, torch.zeros_like(confidence))

    # remove borders
    border = radius
    confidence[:, :, :border] = 0.
    confidence[:, :, -border:] = 0.
    confidence[:, :, :, :border] = 0.
    confidence[:, :, :, -border:] = 0.

    # confidence topk
    _, index = confidence.flatten(1).topk(topk, dim=1, largest=True, sorted=True)

    index_v = torch.div(index, confidence.size(-1) , rounding_mode='trunc')
    index_u = index % confidence.size(-1)
    # back to original index
    index_v += start_H

    return torch.cat([index_u.unsqueeze(-1),index_v.unsqueeze(-1)],dim=-1)

def mask_features_by_keypoints(feature_map, keypoints):
    """
    根据给定的2D关键点坐标，将特征图中非关键点位置的特征置为0。

    参数:
    - feature_map (torch.Tensor): 形状为 [B, C, H, W] 的特征图。
    - keypoints (torch.Tensor): 形状为 [B, N, 2] 的关键点坐标张量，
                                 其中每个点的坐标格式为 (u, v)，即 (x, y) 或 (宽, 高)。

    返回:
    - torch.Tensor: 形状为 [B, C, H, W] 的处理后的特征图。
    """
    # 1. 获取输入张量的尺寸信息
    B, C, H, W = feature_map.shape
    N = keypoints.shape[1]

    # 2. 创建一个全为0的空白蒙版
    # 蒙版的空间维度应与特征图一致
    mask = torch.zeros(B, H, W, device=feature_map.device, dtype=torch.float32)

    # 3. 准备用于高级索引的坐标
    # 将浮点坐标转换为长整型，以便用作索引
    # keypoints[..., 0] 是 u (宽度, W维度), keypoints[..., 1] 是 v (高度, H维度)
    u_coords = keypoints[..., 0].long()
    v_coords = keypoints[..., 1].long()

    # 创建批次索引，形状为 [B, N]，值为 [[0,0...], [1,1...], ...]
    # 这确保每个点的索引都与其所属的批次对应
    batch_indices = torch.arange(B, device=feature_map.device).unsqueeze(1).expand(-1, N)

    # 4. 使用高级索引，将关键点位置“点亮”为1
    # mask[批次索引, 高度索引, 宽度索引] = 1
    # 注意PyTorch索引顺序是 (B, H, W)，所以v坐标在前，u坐标在后
    mask[batch_indices, v_coords, u_coords] = 1.0

    # 5. 将蒙版应用于特征图
    # a. 将蒙版 [B, H, W] 扩展为 [B, 1, H, W] 以匹配特征图的维度
    mask = mask.unsqueeze(1)
    
    # b. 使用广播机制将蒙版与特征图相乘
    # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
    masked_feature_map = feature_map * mask
    
    return masked_feature_map

class Model(nn.Module):
    def __init__(self, args, device=None):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.device = device

        self.level = sorted([int(item) for item in args.level.split('_')])
        self.N_iters = args.N_iters
        self.channels = [64,16,4]

        self.SatFeatureNet = VGGUnet(self.level, self.channels)


        self.GrdFeatureForT = VGGUnet(self.level, self.channels)
        self.SatFeatureForT = VGGUnet(self.level, self.channels)

        self.conf_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.t2ga = T2GA(64)

        self.meters_per_pixel = {}
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel[level] = meter_per_pixel * (2 ** (3 - level))

        self.coe_R = nn.Parameter(torch.tensor(-5., dtype=torch.float32), requires_grad=True)
        self.coe_T = nn.Parameter(torch.tensor(-3., dtype=torch.float32), requires_grad=True)

        self.dino_feat = DINO()
        self.dpt = DPT(self.dino_feat.feat_dim)
        # if self.args.use_uncertainty:
        #     self.uncertain_net = Uncertainty(self.channels)

        self.masks = {}
        for level in range(4):
            A = 512 / 2**(3-level)
            XYZ_1 = self.sat2world(A)  # [ sidelength,sidelength,4]

            B = 1
            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
            heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)

            ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                                          [0.0000, 482.7076, 125.0034],
                                          [0.0000, 0.0000, 1.0000]]],
                                        dtype=torch.float32, requires_grad=True, device=self.device)
            ori_grdH, ori_grdW = 256, 1024
            H, W = ori_grdH, ori_grdW

            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, ori_camera_k, H, W,
                                                       ori_grdH, ori_grdW)
            # [B, H, W, 2], [B, H, W, 1]
            self.masks[level] = mask[:, :, :, 0]

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def initialize_grd_encoder(self):
        self.GrdFeatureNet = copy.deepcopy(self.SatFeatureNet)

    def initialize_grd_encoder_for_T(self):
        self.GrdFeatureForT = copy.deepcopy(self.SatFeatureNet)

    def grd_img2cam(self, grd_H, grd_W, ori_grdH, ori_grdW):

        ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                                      [0.0000, 482.7076, 125.0034],
                                      [0.0000, 0.0000, 1.0000]]],
                                    dtype=torch.float32, requires_grad=True)  # [1, 3, 3]

        camera_height = utils.get_camera_height()

        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, grd_W, dtype=torch.float32))
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0)  # [1, grd_H, grd_W, 3]
        xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

        w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                        utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
        xyz_grd = xyz_w * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates
        # xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)

        mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

        return xyz_grd, mask, xyz_w

    def grd2cam2world2sat(self, ori_shift_u, ori_shift_v, ori_heading, level,
                          satmap_sidelength, require_jac=False, gt_depth=None):
        '''
        realword: X: south, Y:down, Z: east
        camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        Args:
            ori_shift_u: [B, 1]
            ori_shift_v: [B, 1]
            heading: [B, 1]
            XYZ_1: [H,W,4]
            ori_camera_k: [B,3,3]
            grd_H:
            grd_W:
            ori_grdH:
            ori_grdW:

        Returns:
        '''
        B, _ = ori_heading.shape
        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        shift_u = ori_shift_u * self.args.shift_range_lon
        shift_v = ori_shift_v * self.args.shift_range_lat

        cos = torch.cos(heading)
        sin = torch.sin(heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B, 9]
        R = R.view(B, 3, 3)  # shape = [B, N, 3, 3]
        # this R is the inverse of the R in G2SP

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u[:, :1])
        T0 = torch.cat([shift_v, height, -shift_u], dim=-1)  # shape = [B, 3]
        # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
        # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
        T = torch.sum(-R * T0[:, None, :], dim=-1)  # [B, 3]

        # The above R, T define transformation from camera to world

        if self.args.use_gt_depth and gt_depth != None:
            xyz_w = self.xyz_grds[level][2].detach().to(ori_shift_u.device).repeat(B, 1, 1, 1)
            H, W = xyz_w.shape[1:-1]
            depth = F.interpolate(gt_depth[:, None, :, :], (H, W))
            xyz_grd = xyz_w * depth.permute(0, 2, 3, 1)
            mask = (gt_depth != -1).float()
            mask = F.interpolate(mask[:, None, :, :], (H, W), mode='nearest')
            mask = mask[:, 0, :, :]
        else:
            xyz_grd = self.xyz_grds[level][0].detach().to(ori_shift_u.device).repeat(B, 1, 1, 1)
            mask = self.xyz_grds[level][1].detach().to(ori_shift_u.device).repeat(B, 1, 1)  # [B, grd_H, grd_W]
        grd_H, grd_W = xyz_grd.shape[1:3]

        xyz = torch.sum(R[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + T[:, None, None, :]
        # [B, grd_H, grd_W, 3]
        # zx0 = torch.stack([xyz[..., 2], xyz[..., 0]], dim=-1)  # [B, N, grd_H, grd_W, 2]
        R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True) \
            .reshape(2, 3)
        zx = torch.sum(R_sat[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
        # [B, grd_H, grd_W, 2]
        # assert zx == zx0

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        sat_uv = zx / meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv

        if require_jac:
            dR_dtheta = self.args.rotation_range / 180 * np.pi * \
                        torch.cat([-sin, zeros, -cos, zeros, zeros, zeros, cos, zeros, -sin],
                                  dim=-1)  # shape = [B, N, 9]
            dR_dtheta = dR_dtheta.view(B, 3, 3)
            # R_zeros = torch.zeros_like(dR_dtheta)

            dT0_dshiftu = self.args.shift_range_lon * torch.tensor([0., 0., -1.], dtype=torch.float32,
                                                                   device=shift_u.device,
                                                                   requires_grad=True).view(1, 3).repeat(B, 1)
            dT0_dshiftv = self.args.shift_range_lat * torch.tensor([1., 0., 0.], dtype=torch.float32,
                                                                   device=shift_u.device,
                                                                   requires_grad=True).view(1, 3).repeat(B, 1)
            # T0_zeros = torch.zeros_like(dT0_dx)

            dxyz_dshiftu = torch.sum(-R * dT0_dshiftu[:, None, :], dim=-1)[:, None, None, :]. \
                repeat([1, grd_H, grd_W, 1])  # [B, grd_H, grd_W, 3]
            dxyz_dshiftv = torch.sum(-R * dT0_dshiftv[:, None, :], dim=-1)[:, None, None, :]. \
                repeat([1, grd_H, grd_W, 1])  # [B, grd_H, grd_W, 3]
            dxyz_dtheta = torch.sum(dR_dtheta[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + \
                          torch.sum(-dR_dtheta * T0[:, None, :], dim=-1)[:, None, None, :]

            duv_dshiftu = 1 / meter_per_pixel * \
                          torch.sum(R_sat[None, None, None, :, :] * dxyz_dshiftu[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]
            duv_dshiftv = 1 / meter_per_pixel * \
                          torch.sum(R_sat[None, None, None, :, :] * dxyz_dshiftv[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]
            duv_dtheta = 1 / meter_per_pixel * \
                         torch.sum(R_sat[None, None, None, :, :] * dxyz_dtheta[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]

            # duv_dshift = torch.stack([duv_dx, duv_dy], dim=0)
            # duv_dtheta = duv_dtheta.unsqueeze(dim=0)

            return sat_uv, mask, duv_dshiftu, duv_dshiftv, duv_dtheta

        return sat_uv, mask, None, None, None

    def project_map_to_grd(self, sat_f, sat_c, shift_u, shift_v, heading, level, require_jac=True, gt_depth=None):
        '''
        Args:
            sat_f: [B, C, H, W]
            sat_c: [B, 1, H, W]
            shift_u: [B, 2]
            shift_v: [B, 2]
            heading: [B, 1]
            camera_k: [B, 3, 3]

            ori_grdH:
            ori_grdW:

        Returns:

        '''
        B, C, satmap_sidelength, _ = sat_f.size()
        A = satmap_sidelength

        uv, mask, jac_shiftu, jac_shiftv, jac_heading = self.grd2cam2world2sat(shift_u, shift_v, heading, level,
                                                                               satmap_sidelength, require_jac, gt_depth)
        # [B, H, W, 2], [B, H, W], [B, H, W, 2], [B, H, W, 2], [B,H, W, 2]

        B, grd_H, grd_W, _ = uv.shape
        if require_jac:
            jac = torch.stack([jac_shiftu, jac_shiftv, jac_heading], dim=0)  # [3, B, H, W, 2]

            # jac = jac.reshape(3, -1, grd_H, grd_W, 2)
        else:
            jac = None

        # print('==================')
        # print(jac.shape)
        # print('==================')

        sat_f_trans, new_jac = grid_sample(sat_f,
                                           uv,
                                           jac)
        sat_f_trans = sat_f_trans * mask[:, None, :, :]
        if require_jac:
            new_jac = new_jac * mask[None, :, None, :, :]

        if sat_c is not None:
            sat_c_trans, _ = grid_sample(sat_c, uv)
            sat_c_trans = sat_c_trans * mask[:, None, :, :]
        else:
            sat_c_trans = None

        return sat_f_trans, sat_c_trans, new_jac, uv * mask[:, :, :, None], mask

    def sat2world(self, satmap_sidelength):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def World2GrdImgPixCoordinates(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W, ori_grdH,
                             ori_grdW):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
        B = ori_heading.shape[0]
        shift_u_meters = self.args.shift_range_lon * ori_shift_u
        shift_v_meters = self.args.shift_range_lat * ori_shift_v
        heading = ori_heading * self.args.rotation_range / 180 * np.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(B, 3, 3)  # shape = [B,3,3]

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
        # T = torch.einsum('bij, bjk -> bik', R, T0)
        # T = R @ T0

        # P = K[R|T]
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        # P = torch.einsum('bij, bjk -> bik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,3,4]
        P = camera_k @ torch.cat([R, T], dim=-1)

        # uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        H, W = uv.shape[1:-1]
        assert (H == W)

        # with torch.no_grad():
        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6) * \
               torch.greater_equal(uv[:, :, :, 0:1], torch.zeros_like(uv[:, :, :, 0:1])) * \
               torch.less(uv[:, :, :, 0:1], torch.ones_like(uv[:, :, :, 0:1]) * grd_W) * \
               torch.greater_equal(uv[:, :, :, 1:2], torch.zeros_like(uv[:, :, :, 1:2])) * \
               torch.less(uv[:, :, :, 1:2], torch.ones_like(uv[:, :, :, 1:2]) * grd_H)
        uv = uv * mask

        return uv, mask
        # return uv1

    def project_grd_to_map(self, grd_f, grd_c, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH,
                           ori_grdW, require_jac=True):
        # inputs:
        #   grd_f: ground features: B,C,H,W
        #   shift: B, S, 2
        #   heading: heading angle: B,S
        #   camera_k: 3*3 K matrix of left color camera : B*3*3
        # return:
        #   grd_f_trans: B,S,E,C,satmap_sidelength,satmap_sidelength

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength)  # [ sidelength,sidelength,4]

        if self.args.proj == 'geo' or self.args.proj == 'CrossAttn':
            uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k, H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
            # [B, H, W, 2], [B, H, W, 1]

        grd_f_trans, new_jac = grid_sample(grd_f, uv, None)
        # [B,C,sidelength,sidelength], [3, B, C, sidelength, sidelength]
        grd_f_trans = grd_f_trans * mask[:, None, :, :, 0]
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
            grd_c_trans = grd_c_trans * mask[:, None, :, :, 0]
        else:
            grd_c_trans = None


        return grd_f_trans, grd_c_trans, uv, mask

    def inplane_uv(self, ori_shift_u, ori_shift_v, ori_heading, satmap_sidelength):
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength

        B = ori_heading.shape[0]
        shift_u_pixels = self.args.shift_range_lon * ori_shift_u / meter_per_pixel
        shift_v_pixels = self.args.shift_range_lat * ori_shift_v / meter_per_pixel
        T = torch.cat([-shift_u_pixels, shift_v_pixels], dim=-1)  # [B, 2]

        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        R = torch.cat([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)

        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        v, u = torch.meshgrid(i, j)  # i:h,j:w
        uv_2 = torch.stack([u, v], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1).float()  # [B, H, W, 2]
        uv_2 = uv_2 - satmap_sidelength / 2

        uv_1 = torch.einsum('bij, bhwj->bhwi', R, uv_2)
        uv_0 = uv_1 + T[:, None, None, :]  # [B, H, W, 2]

        uv = uv_0 + satmap_sidelength / 2
        return uv

    def Trans_update(self, shift_u, shift_v, heading, grd_feat_proj, sat_feat, level):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat, level)  # [B, 3]
        # print('=======================')
        # print('delta.shape: ', delta.shape)
        # print('shift_u.shape', shift_u.shape)
        # print('=======================')

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = heading + delta[:, 2:3]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        # shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new

    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

        gt_delta_x = - gt_shift_u[:, 0] * self.args.shift_range_lon
        gt_delta_y = - gt_shift_v[:, 0] * self.args.shift_range_lat

        gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
        gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[level]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            # import pdb; pdb.set_trace()
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))

    def weak_supervise_loss(self, corr_maps):
        losses = []
        for corr in corr_maps:
            M, N, H, W = corr.shape
            assert M == N
            dis = torch.max(corr.reshape(M, N, -1), dim=-1)[0]
            pos = torch.diagonal(dis) # [M]
            pos_neg = pos.reshape(-1, 1) - dis
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))


    def NeuralOptimizer(self, grd_feat_dict, sat_feat_dict, B, left_camera_k=None, ori_grdH=None, ori_grdW=None):

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)

        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in self.level:
                sat_feat = sat_feat_dict[level]
                grd_feat = grd_feat_dict[level]

                if self.args.stage == 0:
                    uv = self.inplane_uv(shift_u, shift_v, heading, sat_feat.shape[-1])
                    overhead_feat, _ = grid_sample(
                        grd_feat * self.masks[level].clone()[:, None, :, :].repeat(B, 1, 1, 1),
                        uv, jac=None)
                    # single_features_to_RGB(overhead_feat)
                elif self.args.stage > 0:
                    A = sat_feat.shape[-1]
                    overhead_feat, _, _, _ = self.project_grd_to_map(
                        grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

                shift_u_new, shift_v_new, heading_new = self.Trans_update(
                    shift_u, shift_v, heading, overhead_feat, sat_feat, level)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        return shift_lats, shift_lons, thetas

    def forward(self, sat_align_cam, sat_map, grd_img_left, grd_depth, grd_ori, left_camera_k, gt_heading=None, gt_shift_u=None, gt_shift_v=None, train=False, loop=None, save_dir=None):
        '''
        rot_corr
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''

        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')
        # sat_align_cam_ = transforms.ToPILImage()(sat_align_cam[0])
        # sat_align_cam_.save('sat_align_cam.png')
        #
        # uv = self.inplane_uv(gt_shift_u, gt_shift_v, gt_heading, sat_map.shape[-1])
        # sat_align_cam_trans, _ = grid_sample(
        #     sat_align_cam,
        #     uv, jac=None)
        # sat_align_cam_trans = transforms.ToPILImage()(sat_align_cam_trans[0])
        # sat_align_cam_trans.save('sat_align_cam_trans.png')


        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        g2s_feat_dict = {}
        g2s_conf_dict = {}

        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        thetas = heading.unsqueeze(1)
        shift_lats = None
        shift_lons = None

        # ----------------- Translation Stage ---------------------------


        grd_feat_dict_forT, grd_conf_dict_forT = self.GrdFeatureForT(grd_img_left)
        sat_feat_dict_forT, sat_conf_dict_forT = self.SatFeatureForT(sat_map)

        level = self.level[0]

        grd_uv_dict = {}
        mask_dict = {}

        # meter_per_pixel = self.meters_per_pixel[level]
        sat_feat = sat_feat_dict_forT[level]
        grd_feat = grd_feat_dict_forT[level]

        grd_feat = self.t2ga(grd_feat)
        grd_conf = nn.Sigmoid()(-self.conf_net(grd_feat))

        p2d_grd_key = extract_keypoints(grd_conf)

        grd_feat_key = mask_features_by_keypoints(grd_feat, p2d_grd_key)
        grd_conf_key = mask_features_by_keypoints(grd_conf, p2d_grd_key)
        A = sat_feat.shape[-1]

        grd_feat_proj, grd_conf_proj, grd_uv, mask = self.project_grd_to_map(
            grd_feat_key, grd_conf_key, shift_u, shift_v, heading, left_camera_k, A, ori_grdH,
            ori_grdW,
            require_jac=False)

        g2s_feat_dict[level] = grd_feat_proj
        g2s_conf_dict[level] = grd_conf_proj
        grd_uv_dict[level] = grd_uv
        mask_dict[level] = mask

        for _, level in enumerate(self.level):

            meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_dict_forT[level]

            A = sat_feat.shape[-1]

            crop_H = int(A - 20 * 3 / meter_per_pixel)
            crop_W = int(A - 20 * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])

            g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])

            g2s_feat_dict[level] = g2s_feat
            g2s_conf_dict[level] = (g2s_feat != 0).any(dim=1, keepdim=True).float()

        render_loss = torch.tensor(0.0).to(self.device)
        return sat_feat_dict_forT, sat_conf_dict_forT, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss



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


def weak_supervise_loss(corr_maps):
    '''
    triplet loss/ metric learning loss for self-supervision
    corr_maps: dict
    key -- level; value -- corr map
    '''
    losses = []
    for key, corr in corr_maps:
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        losses.append(loss)

    return torch.mean(torch.stack(losses, dim=0))


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meter_per_pixels, GPS_error=5):
    '''
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = []
    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos
    # ------------------------------------------------

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

        # ---------- preparing for GPS error Loss -------
        meter_per_pixel = meter_per_pixels[level]
        w = (torch.round(W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)).long() # [B]
        h = (torch.round(H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)).long() # [B]
        radius = int(np.ceil(GPS_error / meter_per_pixel))
        GPS_dis = []
        for b_idx in range(M):
            # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
            start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius)
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius)
            start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius)
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius)
            GPS_dis.append(torch.min(
                corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))


def GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, meters_per_pixel):
    '''
    Used when GT GPS lables are highly reliable.
    This function does not handle the rotation issue.
    '''
    levels = [int(item) for item in args.level.split('_')]

    # cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    # sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    #
    # gt_delta_x = gt_shift_u[:, 0] * args.shift_range_lon
    # gt_delta_y = gt_shift_v[:, 0] * args.shift_range_lat
    #
    # gt_delta_x_rot = - gt_delta_x * cos - gt_delta_y * sin
    # gt_delta_y_rot = gt_delta_x * sin - gt_delta_y * cos

    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

    gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat

    gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

    losses = []
    # for level in range(len(corr_maps)):
    for _, level in enumerate(levels):
        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        meter_per_pixel = meters_per_pixel[level]

        w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
        h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

        pos = corr[range(B), h.long(), w.long()]  # [B]
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.sum(torch.stack(losses, dim=0))


def corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, meter_per_pixels, gt_heading, masks=None):
    '''
    to be used during inference
    '''

    level = max([int(item) for item in args.level.split('_')])
    meter_per_pixel = meter_per_pixels[level]

    sat_feat = sat_feat_dict[level]
    sat_conf = sat_conf_dict[level]
    g2s_feat = g2s_feat_dict[level]
    g2s_conf = g2s_conf_dict[level]

    B, C, crop_H, crop_W = g2s_feat.shape
    A = sat_feat.shape[2]

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

        mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
        l2_norm_kernel = mask.repeat(1, C, 1, 1)
        sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)[0]
        denominator_sat = torch.maximum(torch.sqrt(sat_feat_squared_sum + 1e-8), torch.ones_like(sat_feat_squared_sum) * 1e-6)  # 滑动窗口的 L2 范数
        # denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
        # denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))
        
        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        # denominator = corr / denominator_sat / denominator_grd

    denominator = denominator_sat * denominator_grd

    denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

    corr = corr / denominator  # [B, H, W]

    corr_H = int(args.shift_range_lat * 3 / meter_per_pixel)
    corr_W = int(args.shift_range_lon * 3 / meter_per_pixel)

    corr = TF.center_crop(corr[:, None], [corr_H, corr_W])[:, 0]

    B, corr_H, corr_W = corr.shape

    max_index = torch.argmax(corr.reshape(B, -1), dim=1)

    if args.visualize:
        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * np.power(2, 3 - level)
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * np.power(2, 3 - level)
        return pred_u, pred_v, corr

    else:

        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lon
        pred_v = -(max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lat

        cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

        pred_u1 = pred_u * cos + pred_v * sin
        pred_v1 = - pred_u * sin + pred_v * cos

        return pred_u1, pred_v1, corr



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




def loss_func(shift_lats, shift_lons, thetas,
              gt_shift_lat, gt_shift_lon, gt_theta,
              coe_shift_lat=100, coe_shift_lon=100, coe_theta=100):
    '''
    Args:
        loss_method:
        ref_feat_list:
        pred_feat_dict:
        gt_feat_dict:
        shift_lats: [B, N_iters, Level]
        shift_lons: [B, N_iters, Level]
        thetas: [B, N_iters, Level]
        gt_shift_lat: [B]
        gt_shift_lon: [B]
        gt_theta: [B]
        pred_uv_dict:
        gt_uv_dict:
        coe_shift_lat:
        coe_shift_lon:
        coe_theta:
        coe_L1:
        coe_L2:
        coe_L3:
        coe_L4:

    Returns:

    '''

    shift_lat_delta0 = torch.abs(shift_lats - gt_shift_lat[:, None, None])  # [B, N_iters, Level]
    shift_lon_delta0 = torch.abs(shift_lons - gt_shift_lon[:, None, None])  # [B, N_iters, Level]
    thetas_delta0 = torch.abs(thetas - gt_theta[:, None, None])  # [B, N_iters, level]

    shift_lat_delta = torch.mean(shift_lat_delta0, dim=0)  # [N_iters, Level]
    shift_lon_delta = torch.mean(shift_lon_delta0, dim=0)  # [N_iters, Level]
    thetas_delta = torch.mean(thetas_delta0, dim=0)  # [N_iters, level]

    shift_lat_decrease = shift_lat_delta[0, 0] - shift_lat_delta[-1, -1]  # scalar
    shift_lon_decrease = shift_lon_delta[0, 0] - shift_lon_delta[-1, -1]  # scalar
    thetas_decrease = thetas_delta[0, 0] - thetas_delta[-1, -1]  # scalar

    losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
    loss_decrease = losses[0, 0] - losses[-1, -1]  # scalar
    loss = torch.mean(losses)  # mean or sum
    loss_last = losses[-1]

    return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
        shift_lat_delta[-1, -1], shift_lon_delta[-1, -1], thetas_delta[-1, -1]

