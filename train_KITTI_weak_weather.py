import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
from dataLoader.KITTI_dataset_weather import load_train_data, load_test1_data, load_test2_data
import scipy.io as scio
from torchvision import transforms
import ssl
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib.cm as cm # 导入 colormap 模块
import matplotlib.colors as mcolors # 导入 colors 模块

to_pil_img = transforms.ToPILImage()
ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

# from models_ford import loss_func, loss_func_l2
from models.models_kitti_nips import Model, batch_wise_cross_corr, corr_for_translation, weak_supervise_loss, \
    Weakly_supervised_loss_w_GPS_error, corr_for_accurate_translation_supervision, GT_triplet_loss, loss_func

import numpy as np
import os
import argparse
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


def show_cam_on_image(img: np.ndarray,
                         mask: np.ndarray,
                         use_rgb: bool = True, # Still useful if img input format varies
                         colormap: str = 'viridis') -> np.ndarray:
    """ This function overlays the cam mask on the image as a heatmap using Matplotlib colormaps.

    :param img: The base image in RGB or BGR format, expected as np.float32 in the range [0, 1].
    :param mask: The 2D cam mask (grayscale heatmap data).
    :param use_rgb: Whether the input image `img` is in RGB format (affects potential blending interpretation, though addition is channel-wise).
    :param colormap: The name of the Matplotlib colormap to be used (e.g., 'viridis', 'jet', 'coolwarm_r', etc.)
                     or a Colormap object itself.
    :returns: The blended image with the cam overlay as np.uint8 in the same format (RGB/BGR) as input img.
    """
    # --- Input Validation ---
    if np.max(img) > 1.001 or np.min(img) < -0.001: # Allow for small float inaccuracies
         # Try to normalize if it looks like 0-255 range
         if np.max(img) > 2.0 and np.min(img) >=0:
             print("Warning: Input image seems to be in [0, 255] range. Normalizing to [0, 1].")
             img = img.astype(np.float32) / 255.0
         else:
            raise ValueError("Input image `img` should be np.float32 in the range [0, 1]")
    if mask.ndim != 2:
        raise ValueError(f"Input mask must be 2D, but got shape {mask.shape}")

    # --- Mask Normalization [0, 1] ---
    mask_min = np.min(mask)
    mask_max = np.max(mask)
    if mask_max == mask_min:
        # Handle constant mask: make it fully transparent or a mid-value gray?
        # Option 1: Make it fully transparent equivalent (0)
        # normalized_mask = np.zeros_like(mask, dtype=np.float32)
        # Option 2: Map to a mid-value (0.5) - better visual if value isn't zero
        normalized_mask = np.full_like(mask, 0.5, dtype=np.float32)
        print("Warning: Input mask is constant.")
    else:
        normalized_mask = (mask - mask_min) / (mask_max - mask_min)

    # --- Apply Matplotlib Colormap ---
    try:
        cmap = plt.get_cmap(colormap)
        # Apply colormap: cmap returns RGBA values in range [0, 1]
        heatmap_rgba = cmap(normalized_mask)
        # Keep only RGB channels
        heatmap_rgb = heatmap_rgba[:, :, :3] # Shape: [H, W, 3], range [0, 1], float32
    except ValueError:
        print(f"Warning: Colormap '{colormap}' not found. Using 'viridis'.")
        cmap = plt.get_cmap('viridis')
        heatmap_rgba = cmap(normalized_mask)
        heatmap_rgb = heatmap_rgba[:, :, :3]

    # --- Blending ---
    # Ensure heatmap_rgb and img are float32 [0, 1]
    heatmap_float = heatmap_rgb.astype(np.float32)
    img_float = img.astype(np.float32)

    # Choose blending method:
    # Option A: Simple addition (like original if scale was 0.5)
    # cam = heatmap_float + img_float
    # Option B: Weighted averaging (often preferred)
    alpha = 0.5 # Adjust transparency of heatmap
    cam = alpha * heatmap_float + (1 - alpha) * img_float

    # --- Final Normalization and Conversion ---
    # Normalize the blended image to be in [0, 1] by clipping or dividing by max
    # Clipping is often safer to preserve relative brightness
    cam = np.clip(cam, 0, 1)
    # cam = cam / np.max(cam) # Alternative: scales relative brightness

    # Convert to uint8 in the range [0, 255]
    cam_uint8 = np.uint8(255 * cam)

    # --- Ensure output format matches input 'use_rgb' flag ---
    # (Matplotlib output is RGB, so convert *back* to BGR if use_rgb is False)
    # This step is only needed if the calling code strictly expects BGR based on use_rgb=False
    # if not use_rgb:
    #     cam_uint8 = cv2.cvtColor(cam_uint8, cv2.COLOR_RGB2BGR)

    return cam_uint8

def test1_orien(net_test, args, save_path, epoch):
    
    net_test.eval()

    dataloader = load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))
    
    pred_oriens = []

    gt_oriens = []

    start_time = time.time()

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, grd_depth = [item.to(device) for
                                                                                                        item in Data[:8]]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss = \
                net(sat_align_cam, sat_map, grd_left_imgs, grd_depth, left_camera_k, gt_heading)

            pred_orien = thetas[:, -1, -1]
            pred_oriens.append(pred_orien.data.cpu().numpy() * args.rotation_range)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)
            
            if i % 20 == 0:
                print(i)
    
    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size
    pred_oriens = np.concatenate(pred_oriens, axis=0)
    
    gt_oriens = np.concatenate(gt_oriens, axis=0)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_angle = np.abs(gt_oriens)
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'test1_results_orien.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)
    line = 'Test1_orien results:'
    print(line)
    f.write(line + '\n')    
    line = 'Angle average (init, pred): ' + str(np.mean(np.abs(gt_oriens))) + ' ' + str(np.mean(angle_diff))
    print(line)
    f.write(line + '\n')
    line = 'Angle median (init, pred): ' + str(np.median(np.abs(gt_oriens))) + ' ' + str(np.median(angle_diff))
    print(line)
    f.write(line + '\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')
    f.close()

    net_test.train()

    return

def test2_orien(net_test, args, save_path, epoch):
    
    net_test.eval()

    dataloader = load_test2_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))

    pred_oriens = []

    gt_oriens = []

    start_time = time.time()

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, grd_depth = [item.to(device) for
                                                                                                        item in Data[:8]]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss = \
                net(sat_align_cam, sat_map, grd_left_imgs, grd_depth, left_camera_k, gt_heading)

            pred_orien = thetas[:, -1, -1]
            pred_oriens.append(pred_orien.data.cpu().numpy() * args.rotation_range)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)
            
            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size
    pred_oriens = np.concatenate(pred_oriens, axis=0)
    
    gt_oriens = np.concatenate(gt_oriens, axis=0)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_angle = np.abs(gt_oriens)
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'test2_results_orien.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)
    line = 'Test2_orien results:'
    print(line)
    f.write(line + '\n')    
    line = 'Angle average (init, pred): ' + str(np.mean(np.abs(gt_oriens))) + ' ' + str(np.mean(angle_diff))
    print(line)
    f.write(line + '\n')
    line = 'Angle median (init, pred): ' + str(np.median(np.abs(gt_oriens))) + ' ' + str(np.median(angle_diff))
    print(line)
    f.write(line + '\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')
    f.close()

    net_test.train()

    return

def test1(net_test, args, save_path, epoch):

    net_test.eval()

    dataloader = load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))
    pred_lons = []
    pred_lats = []
    pred_oriens = []

    pred_lons_neuralOpt = []
    pred_lats_neuralOpt = []
    
    gt_lons = []
    gt_lats = []
    gt_oriens = []

    start_time = time.time()

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, grd_left_imgs_ori, gt_shift_u, gt_shift_v, gt_heading, grd_depth = [item.to(device) for
                                                                                                        item in Data[:9]]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss = \
                net(sat_align_cam, sat_map, grd_left_imgs, grd_depth, grd_left_imgs_ori, left_camera_k, gt_heading)

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args,
                                                        net_test.meters_per_pixel,
                                                        gt_heading=gt_heading,
                                                        masks=mask_dict)
            
            pred_orien = thetas[:, -1, -1]
            pred_angle = pred_orien.data.cpu().numpy() * args.rotation_range / 180 * np.pi

            # gt heading here is just to decompose the pred_u & pred_v in the lateral and longitudinal direction
            # for evaluation purpose only

            if args.visualize:

                # 示例处理
                idx = 0
                cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
                sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
                gt_delta_x = - gt_shift_u[:, 0] * args.shift_range_lon
                gt_delta_y = - gt_shift_v[:, 0] * args.shift_range_lat
                gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
                gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos
                gt_u1 = torch.round(gt_delta_x_rot / net_test.meters_per_pixel[3]).data.cpu().numpy()
                gt_v1 = torch.round(gt_delta_y_rot / net_test.meters_per_pixel[3]).data.cpu().numpy()
                gt_angle = gt_heading[:, 0].data.cpu().numpy() * args.rotation_range / 180 * np.pi

                prob_map = np.asarray(Image.fromarray(corr[idx].data.cpu().numpy()).resize((corr.shape[2]*4, corr.shape[1]*4)))
                img = sat_map[idx].permute(1, 2, 0).data.cpu().numpy()[
                    (512 - prob_map.shape[0]) // 2: -(512 - prob_map.shape[0]) // 2,
                    (512 - corr.shape[2]*4) // 2: -(512 - corr.shape[2]*4) // 2, :]
                cmap_name = 'rainbow'
                
                overlay = show_cam_on_image(img, prob_map, False, cmap_name)

                # 创建绘图
                fig, ax = plt.subplots()
                # plt.subplots_adjust(top=0.6)  # 留白空间可以根据需要调整，top的值控制上方空白的大小

                A = overlay.shape[0]
                
                # # 1) 用不可见的方式画出热力图，本质是为了生成 colorbar
                # im_for_cbar = ax.imshow(prob_map, cmap='hsv', alpha=0.0)  
                # # 这里 alpha=0 让它不覆盖图像，但 Matplotlib 仍然知道它的数据范围

                # # 2) 放置 colorbar（与上面的 im_for_cbar 关联）
                # cbar = plt.colorbar(im_for_cbar, ax=ax)
                # cbar.set_label("Activation / Heat", color='black')  # 可选：给 colorbar 命名
                
                # 显示图像
                shw = ax.imshow(overlay)
                # im = ax.imshow(img)


                # --- 添加颜色条1 ---
                # 1. 创建一个 ScalarMappable 对象，它知道数值和颜色的映射关系
                #    我们需要 prob_map_resized 的最小值和最大值来设定范围
                vmin = prob_map.min()
                vmax = prob_map.max()
                #    选择与 show_cam_on_image 中使用的 colormap 匹配的 matplotlib colormap
                cmap = plt.get_cmap(cmap_name) # Matplotlib equivalent of cv2.COLORMAP_HSV
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

                # 2. 添加颜色条到图像
                #    'shw' (the imshow object) can sometimes be used directly if it retained value info,
                #    but using a separate mappable based on the original data is safer.
                cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04) # Adjust fraction and pad as needed
                cbar.set_label('Location probability map', fontsize= 16) # 设置颜色条标签


                ax.axis('off')

                # 绘制 scatter 点：初始化位置、预测位置、GT位置
                # init = ax.scatter(162, 26, color='blue', linewidth=2, edgecolor="w", s=80, zorder=2, marker='D', alpha=0.8)
                pred = ax.scatter(
                    int(pred_u[idx]) + A / 2, 
                    int(pred_v[idx]) + A / 2, 
                    color='green', 
                    s=200, 
                    edgecolor='w', 
                    marker='o', 
                    alpha=0.8
                )
                gt = ax.scatter(
                    int(gt_u1[idx]) + A / 2, 
                    int(gt_v1[idx]) + A / 2, 
                    color='red', 
                    s=400, 
                    edgecolor='w', 
                    marker='$⚑$', 
                    alpha=0.8
                )

                # 添加 legend，调整 legend 的位置到上方留白区域
                # ax.legend(
                #     (init, pred, gt),
                #     ('Weekly', 'Ours', 'GT'),
                #     markerscale=1.2,
                #     frameon=False,
                #     fontsize=12,
                #     edgecolor="black",
                #     labelcolor='black',
                #     shadow=True,
                #     facecolor='w',
                #     loc='upper center',  # 让 legend 在上方居中
                #     bbox_to_anchor=(0.5, 1.15),  # 设置 legend 位置，1.05 表示在图像上方
                #     ncol=3
                # )

                ax.legend(
                    (pred, gt),
                    ('Ours', 'GT'),
                    markerscale=1.2,
                    frameon=False,
                    fontsize=16,
                    edgecolor="black",
                    labelcolor='black',
                    shadow=True,
                    facecolor='w',
                    loc='upper center',  # 让 legend 在上方居中
                    bbox_to_anchor=(0.5, 1.14),  # 设置 legend 位置，1.05 表示在图像上方
                    ncol=3
                )

                # 绘制 quiver 箭头
                # ax.quiver(66, 186,
                #         np.cos(pred_angle[idx]), np.sin(pred_angle[idx]),
                #         color='w', scale=15, zorder=2)

                # ax.quiver(int(pred_u[idx]) + A / 2, int(pred_v[idx]) + A / 2,
                #         np.cos(pred_angle[idx]), np.sin(pred_angle[idx]),
                #         color='w', scale=15, zorder=2)

                # ax.quiver(int(gt_u1[idx]) + A / 2, int(gt_v1[idx]) + A / 2,
                #         np.cos(gt_angle[idx]), np.sin(gt_angle[idx]),
                #         color='w', scale=15, zorder=2)

                # 添加颜色条
                # plt.colorbar(im, ax=ax)

                # 保存图像
                plt.axis('off')  # 关闭坐标轴
                plt.savefig(
                    'seq/corr.png',
                    transparent=True, dpi=150, bbox_inches='tight', pad_inches=0.5)
                plt.close()


            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())
            pred_oriens.append(pred_orien.data.cpu().numpy() * args.rotation_range)

            # pred_lons_neuralOpt.append(shift_lons[:, -1, -1].data.cpu().numpy())
            # pred_lats_neuralOpt.append(shift_lats[:, -1, -1].data.cpu().numpy())

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)


            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)
    pred_oriens = np.concatenate(pred_oriens, axis=0)

    # pred_lons_neuralOpt = np.concatenate(pred_lons_neuralOpt, axis=0)
    # pred_lats_neuralOpt = np.concatenate(pred_lats_neuralOpt, axis=0)
    
    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    gt_oriens = np.concatenate(gt_oriens, axis=0)

    scio.savemat(os.path.join(save_path, 'test1_result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 'gt_oriens': gt_oriens,
                                                         'pred_lats': pred_lats, 'pred_lons': pred_lons, 'pred_oriens': pred_oriens})

    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)  # [N]
    # distanc_neuralOpt = np.sqrt((pred_lons_neuralOpt - gt_lons) ** 2 + (pred_lats_neuralOpt - gt_lats) ** 2)  # [N]

    init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)
    
    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)

    # diff_lats_neuralOpt = np.abs(pred_lats_neuralOpt - gt_lats)
    # diff_lons_neuralOpt = np.abs(pred_lons_neuralOpt - gt_lons)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_angle = np.abs(gt_oriens)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'test1_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)
    line = 'Test1 results:'
    print(line)
    f.write(line + '\n')

    line = 'Distance average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred by corr, pred by neuralOpt)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')

    line = 'Lateral average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(np.abs(gt_lats))) + ' ' + str(np.mean(diff_lats))
    print(line)
    f.write(line + '\n')
    line = 'Lateral median: (init, pred by corr, pred by neuralOpt)' + str(np.median(np.abs(gt_lats))) + ' ' + str(np.median(diff_lats))
    print(line)
    f.write(line + '\n')

    line = 'Longitudinal average: (init by corr, pred, pred by neuralOpt)' + str(np.mean(np.abs(gt_lons))) + ' ' + str(np.mean(diff_lons))
    print(line)
    f.write(line + '\n')
    line = 'Longitudinal median: (init by corr, pred, pred by neuralOpt)' + str(np.median(np.abs(gt_lons))) + ' ' + str(np.median(diff_lons))
    print(line)
    f.write(line + '\n')

    line = 'Angle average (init, pred): ' + str(np.mean(np.abs(gt_oriens))) + ' ' + str(np.mean(angle_diff))
    print(line)
    f.write(line + '\n')
    line = 'Angle median (init, pred): ' + str(np.median(np.abs(gt_oriens))) + ' ' + str(np.median(angle_diff))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100
        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lats < metrics[idx]) / diff_lats.shape[0] * 100
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100
        line = 'lateral within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    for idx in range(len(metrics)):

        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100
        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    
    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred by corr, pred by neuralOpt): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')
    f.close()

    net_test.train()

    return


def test2(net_test, args, save_path, epoch):

    net_test.eval()

    dataloader = load_test2_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))
    
    pred_lons = []
    pred_lats = []
    pred_oriens = []

    pred_lons_neuralOpt = []
    pred_lats_neuralOpt = []

    gt_lons = []
    gt_lats = []
    gt_oriens = []

    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, grd_left_imgs_ori, gt_shift_u, gt_shift_v, gt_heading, grd_depth = [item.to(device)
                                                                                                        for
                                                                                                        item in
                                                                                                        Data[:9]]
            # if args.stage == 0:
            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss = \
                net(sat_align_cam, sat_map, grd_left_imgs, grd_depth, grd_left_imgs_ori, left_camera_k, gt_heading)
            pred_orien = thetas[:, -1, -1]
            # else:
            #     sat_feat_dict, sat_uncer_dict, g2s_feat_dict, g2s_conf_dict, shift_lats, shift_lons, pred_orien = \
            #         net(sat_align_cam, sat_map, grd_left_imgs, left_camera_k, gt_heading)
            #     pred_orien = pred_orien[:, 0]

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args,
                                                        net_test.meters_per_pixel,
                                                        gt_heading=gt_heading,
                                                        masks=mask_dict)
            # gt heading here is just to decompose the pred_u & pred_v in the lateral and longitudinal direction
            # for evaluation purpose only

            # pred_orien = thetas[:, -1, -1]
            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())
            pred_oriens.append(pred_orien.data.cpu().numpy() * args.rotation_range)

            # pred_lons_neuralOpt.append(shift_lons[:, -1, -1].data.cpu().numpy())
            # pred_lats_neuralOpt.append(shift_lats[:, -1, -1].data.cpu().numpy())

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)

            if i % 20 == 0:
                print(i)

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)
    pred_oriens = np.concatenate(pred_oriens, axis=0)

    # pred_lons_neuralOpt = np.concatenate(pred_lons_neuralOpt, axis=0)
    # pred_lats_neuralOpt = np.concatenate(pred_lats_neuralOpt, axis=0)

    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    gt_oriens = np.concatenate(gt_oriens, axis=0)

    scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 'gt_oriens': gt_oriens, 
                                                         'pred_lats': pred_lats, 'pred_lons': pred_lons, 'pred_oriens': pred_oriens})
    # scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 
    #                                                      'pred_lats': pred_lats, 'pred_lons': pred_lons})

    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)  # [N]
    # distanc_neuralOpt = np.sqrt((pred_lons_neuralOpt - gt_lons) ** 2 + (pred_lats_neuralOpt - gt_lats) ** 2)  # [N]

    init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)

    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)

    # diff_lats_neuralOpt = np.abs(pred_lats_neuralOpt - gt_lats)
    # diff_lons_neuralOpt = np.abs(pred_lons_neuralOpt - gt_lons)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
  
    init_angle = np.abs(gt_oriens)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'test2_results.txt'), 'a')
    print('-------------------------')
    f.write('------------------------\n')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))

    line = 'Test2 results:'
    print(line)
    f.write(line + '\n')
    
    line = 'Distance average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred by corr, pred by neuralOpt)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')

    line = 'Lateral average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(np.abs(gt_lats))) + ' ' + str(np.mean(diff_lats))
    print(line)
    f.write(line + '\n')
    line = 'Lateral median: (init, pred by corr, pred by neuralOpt)' + str(np.median(np.abs(gt_lats))) + ' ' + str(np.median(diff_lats))
    print(line)
    f.write(line + '\n')

    line = 'Longitudinal average: (init, pred by corr, pred by neuralOpt)' + str(np.mean(np.abs(gt_lons))) + ' ' + str(np.mean(diff_lons))
    print(line)
    f.write(line + '\n')
    line = 'Longitudinal median: (init, pred by corr, pred by neuralOpt)' + str(np.median(np.abs(gt_lons))) + ' ' + str(np.median(diff_lons))
    print(line)
    f.write(line + '\n')

    line = 'Angle average (init, pred): ' + str(np.mean(np.abs(gt_oriens))) + ' ' + str(np.mean(angle_diff))
    print(line)
    f.write(line + '\n')
    line = 'Angle median (init, pred): ' + str(np.median(np.abs(gt_oriens))) + ' ' + str(np.median(angle_diff))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100
        # pred_opt = np.sum(distanc_neuralOpt < metrics[idx]) / distanc_neuralOpt.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred) + ' '
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lats < metrics[idx]) / diff_lats.shape[0] * 100
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100
        # pred_opt = np.sum(diff_lats_neuralOpt < metrics[idx]) / diff_lats_neuralOpt.shape[0] * 100

        line = 'lateral within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred) + ' '
        print(line)
        f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100
        # pred_opt = np.sum(diff_lons_neuralOpt < metrics[idx]) / diff_lons_neuralOpt.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred) + ' '
        print(line)
        f.write(line + '\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred by corr, pred by neuralOpt): ' + str(
            init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.sum((diff_lats < metrics[0])) / diff_lats.shape[0] * 100

    net_test.train()
    return result


def train(net, args, save_path, name_path):
    bestRankResult = 0.0
    
    # optimizer = optim.Adam(net.parameters(), lr=base_lr)
    if args.stage == 0:
        if args.rotation_range == 0:
            params = net.SatFeatureNet.parameters()
        else:
            # params = list(net.SatFeatureNet.parameters()) + list(net.TransRefine.parameters())
            parmas = net.FeatureForT.parameters()
        optimizer = optim.Adam(params, lr=1e-4)

    elif args.stage == 4:
        params = list(net.feat_gaussian_encoder.parameters()) + list(net.dpt.parameters())
        # params = list(net.feat_gaussian_encoder.parameters()) + list(net.FeatureForT.parameters())
        optimizer = optim.AdamW(params, lr= 6.25e-5, weight_decay=5e-3, eps=1e-8)
        # TODO: change strategy to linear
        scale = float(args.batch_size / 8)
        scheduler = OneCycleLR(optimizer, 
                                max_lr=6.25e-5,  # 6.25e-5
                                steps_per_epoch=int(2456 / scale), 
                                epochs=args.epochs, # 5
                                anneal_strategy='cos',
                                pct_start=0.05, # 0.005
                                cycle_momentum=False,
                                )
        # scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer,
        #     1 / 2000,
        #     1,
        #     total_iters=2000,
        # )
    
    time_start = time.time()
    for epoch in range(args.resume, args.epochs):
        net.train()
        trainloader = load_train_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range,
                                      weak_supervise=True, train_noisy=False, stage=args.stage,
                                      data_amount=args.supervise_amount)

        print('batch_size:', args.batch_size, '\n num of batches:', len(trainloader))
        
        global_step = epoch * len(trainloader) * args.batch_size
        
        for Loop, Data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, grd_left_imgs_ori, gt_shift_u, gt_shift_v, gt_heading, grd_depth = [item.to(device) for item in Data[:9]]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, mask_dict, shift_lats, shift_lons, thetas, render_loss = \
                net(sat_align_cam, sat_map, grd_left_imgs, grd_depth, grd_left_imgs_ori, left_camera_k, gt_heading, gt_shift_u, gt_shift_v, loop=Loop, save_dir=save_path)

            if args.stage == 0:
                opt_loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                    shift_lat_last, shift_lon_last, theta_last, \
                    = loss_func(shift_lats, shift_lons, thetas, gt_shift_v[:, 0], gt_shift_u[:, 0], gt_heading[:, 0],
                                torch.exp(-net.coe_R), torch.exp(-net.coe_R), torch.exp(-net.coe_R))


                corr_maps = corr_for_accurate_translation_supervision(sat_feat_dict, sat_conf_dict,
                                                             g2s_feat_dict, g2s_conf_dict, args)

                corr_loss = GT_triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading, args, net.meters_per_pixel)

                if args.rotation_range == 0:
                    loss = corr_loss
                else:
                    loss = opt_loss + \
                           corr_loss * torch.exp(-net.coe_T) + \
                           net.coe_T + net.coe_R

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # This step is responsible for updating weights

                if Loop % 10 == 9:  #
                    time_end = time.time()

                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' DeltaR: ' + str(np.round(thetas_decrease.item(), decimals=2)) +
                          ' FinalR: ' + str(np.round(theta_last.item(), decimals=2)) +
                          ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end

            elif args.stage == 1 or args.stage == 3 or args.stage == 4:
                
                corr_maps = batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, masks=mask_dict)
                corr_loss, GPS_loss = Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v,
                                                                         gt_heading,
                                                                         args,
                                                                         net.meters_per_pixel,
                                                                         args.GPS_error)

                loss = corr_loss + GPS_loss * 0

                R_err = torch.abs(thetas[:, -1, -1].reshape(-1) - gt_heading.reshape(-1)).mean() * args.rotation_range

                # optimizer2.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                # optimizer2.step()
                # 打印每个参数的梯度
                # for name, param in net.named_parameters():
                #     if param.grad is not None:
                #         # 检查梯度张量中是否存在非零元素
                #         if (param.grad != 0).any():
                #             print(name)
                # num_params = sum(p.numel() for p in net.dpt.parameters() if p.requires_grad)
                # print(f"========Number of trainable parameters: {num_params}==========") 
                if Loop % 10 == 9:  #
                    time_end = time.time()
                    current_lr = scheduler.get_last_lr()[0]
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' R error: ' + str(np.round(R_err.item(), decimals=4)) +
                          ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                          ' GPS err loss: ' + str(np.round(GPS_loss.item(), decimals=4)) +
                          f' Learning Rate: {current_lr:.9f}' +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end

        print('Save Model ...')
        if args.stage == 0 or args.stage == 2:
            if not os.path.exists(name_path):
                os.makedirs(name_path)

            torch.save(net.state_dict(), os.path.join(name_path, 'model_' + str(epoch) + '.pth'))
        else:
            if not os.path.exists(name_path):
                os.makedirs(name_path)

            torch.save(net.state_dict(), os.path.join(name_path, 'model_' + str(epoch) + '.pth'))

        if args.stage == 0 or args.stage == 2:
            test1_orien(net, args, name_path, epoch)
            test2_orien(net, args, name_path, epoch)
        else:    
            test1(net, args, name_path, epoch)
            test2(net, args, name_path, epoch)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=6.25e-05, help='learning rate')  # 1e-2

    parser.add_argument('--rotation_range', type=float, default=0., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--level', type=str, default='0_2', help=' ')
    parser.add_argument('--channels', type=str, default='32_16_4', help='64_16_4 ')
    parser.add_argument('--N_iters', type=int, default=1, help='any integer')

    # parser.add_argument('--confidence', type=int, default=0, help='use confidence or not')
    parser.add_argument('--ConfGrd', type=int, default=1, help='use confidence or not for grd image')
    parser.add_argument('--ConfSat', type=int, default=0, help='use confidence or not for sat image')

    parser.add_argument('--share', type=int, default=1, help='share feature extractor for grd and sat or not '
                                                             'in translation estimation')

    parser.add_argument('--Optimizer', type=str, default='TransV1', help='LM or SGD')
    parser.add_argument('--proj', type=str, default='geo', help='geo or CrossAttn')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 1')

    parser.add_argument('--multi_gpu', type=int, default=0, help='0 or 1')

    parser.add_argument('--GPS_error', type=int, default=5, help='')
    parser.add_argument('--GPS_error_coe', type=float, default=0., help='')
    parser.add_argument('--contrastive_coe', type=float, default=0., help='')

    parser.add_argument('--stage', type=int, default=1, help='0 or 1, 0 for self-supervised training, 1 for E2E training')
    parser.add_argument('--task', type=str, default='3DoF',
                        help='')

    parser.add_argument('--supervise_amount', type=float, default=1.0,
                        help='0.1, 0.2, 0.3, ..., 1')
    parser.add_argument('--name', type=str, default='test', help='')
    
    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path= restore_path = '/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage' + str(args.stage) \
                + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range)  \
                + '_Nit' + str(args.N_iters) + '_' + str(args.Optimizer) + '_' + str(args.proj) \
                + '_Level' + args.level + '_Channels' + args.channels

    # if args.ConfGrd and args.stage > 0:
    #     save_path = save_path + '_ConfGrd'
    if args.ConfSat and args.stage > 0:
        save_path = save_path + '_ConfSat'

    if args.GPS_error_coe > 0 and args.stage > 0:

        save_path = save_path + '_GPSerror' + str(args.GPS_error) + '_Coe' + str(args.GPS_error_coe)


    if args.share and args.stage > 0:
        save_path = save_path + '_Share'

    if args.supervise_amount < 1 and args.stage > 0:
        save_path += '_' + str(args.supervise_amount)


    print('save_path:', save_path)
    name_path = save_path + '_' + args.name

    return save_path, restore_path, name_path


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    save_path, restore_path, name_path = getSavePath(args)

    net = Model(args, device=device)

    if args.multi_gpu:
        net = nn.DataParallel(net, dim=0)

    net.to(device)

    if args.test:
        # path = '/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage4/lat20.0m_lon20.0m_rot0.0_Nit1_TransV1_geo_Level1_Channels32_16_4_Share_feat32_offset_0.5_confidence_original_GPS_1e-4/model_9.pth'
        path = '/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage4/lat20.0m_lon20.0m_rot0.0_Nit1_TransV1_geo_Level1_Channels32_16_4_Share_feat32_offset_0.5_confidence_original/model_9.pth'
        
        net.load_state_dict(torch.load(path), strict=False)
        print("resume from " + path)
        # test1(net, args, save_path, epoch=2)
        # test2(net, args, save_path)
        if args.stage == 2:
            test1_orien(net, args, name_path, epoch=0)
            test2_orien(net, args, name_path, epoch=0)
        else:    
            test1(net, args, name_path, epoch=1)
            test2(net, args, name_path, epoch=1)

    else:

        if args.resume:
            path = '/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage4/lat20.0m_lon20.0m_rot0.0_Nit1_TransV1_geo_Level1_Channels32_16_4_Share_op_as_confidence/model_0.pth'
            net.load_state_dict(torch.load(path), strict=False)
            print("resume from " + path)
        
        elif (args.stage == 4) > 0:
            path = '/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage0/lat20.0m_lon20.0m_rot10.0_Nit1_TransV1_geo_Level1_Channels32_16_4_Share_feat32/model_2.pth'
            net.load_state_dict(torch.load(path), strict=False)
            print("load pretrained model from Stage0:")
            print(path)
        
        if args.visualize:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_2.pth')), strict=False)
            print('------------------------')
            print("load pretrained model from ", os.path.join(save_path, 'model_2.pth'))
            print('------------------------')

        lr = args.lr

        train(net, args, save_path, name_path)



def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
