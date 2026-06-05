import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
# from dataLoader.Vigor_dataset import load_vigor_data
from dataLoader.Vigor_dataset_gs import load_vigor_data
from torch.utils.data import Subset
import random

# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio
from torch.optim.lr_scheduler import OneCycleLR

from models.models_vigor import ModelVIGOR, batch_wise_cross_corr, corr_for_translation, Weakly_supervised_loss_w_GPS_error, \
    corr_for_accurate_translation_supervision, GT_triplet_loss
import matplotlib.cm as cm # 导入 colormap 模块
import matplotlib.colors as mcolors # 导入 colors 模块

import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import cv2
from train_KITTI_weak_nips import show_cam_on_image

to_pil_img = transforms.ToPILImage()

def test(net_test, args, save_path):
    ### net evaluation state
    net_test.eval()

    # dataloader = load_vigor_data(args.batch_size, area=args.area)
    dataloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                 train=False, weak_supervise=args.Supervision=='Weakly')

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    start_time = time.time()
    with torch.no_grad():

        for i, Data in enumerate(dataloader, 0):

            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict = \
                net(sat, grd, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage=args.stage, loop=i, save_dir=save_path)

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args, sat_uncer_dict)

            pred_u = pred_u * meter_per_pixel
            pred_v = pred_v * meter_per_pixel

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_shift_u = gt_shift_u * meter_per_pixel * 512 / 4
            gt_shift_v = gt_shift_v * meter_per_pixel * 512 / 4

            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_us': gt_us, 'gt_vs': gt_vs,
                                                         'pred_us': pred_us, 'pred_vs': pred_vs,
                                                         })

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)


    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'results.txt'), 'a')
    # f.write('====================================\n')
    # f.write('       EPOCH: ' + str(epoch) + '\n')
    # print('====================================')
    # print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)

    line = 'Distance average: (init, pred)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    # result = np.mean(distance)

    net_test.train()


def val(dataloader, net, args, save_path, epoch, best=0.0, stage=None):
    time_start = time.time()

    net.eval()
    print('batch_size:', args.batch_size, '\n num of batches:', len(dataloader))

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    start_time = time.time()
    with torch.no_grad():

        for i, Data in enumerate(dataloader, 0):

            grd, sat, depth_imgs, grd_ori, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict = \
                net(sat, grd, depth_imgs, grd_ori, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage=args.stage)

            pred_u, pred_v, corr = corr_for_translation(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict,
                                                        args, sat_uncer_dict)

            if args.visualize:

                visualize_dir = os.path.join(save_path, 'visualization')
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)

                level = max([int(item) for item in args.level.split('_')])
                corr = corr[0]
                corr_H, corr_W = corr.shape

                gt_u1 = gt_shift_u.data.cpu().numpy() * 512/4
                gt_v1 = gt_shift_v.data.cpu().numpy() * 512/4

                visualize_dir = os.path.join(save_path, 'visualization', f"{i}_{0}")
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)
                
                max_index = torch.argmax(corr.reshape(-1)).data.cpu().numpy()

                pred_u = (max_index % corr_W - corr_W / 2) * np.power(2, 3 - level)
                pred_v = (max_index // corr_W - corr_H / 2) * np.power(2, 3 - level)

                prob_map = cv2.resize(corr.data.cpu().numpy(),
                                    (corr.shape[1] * 4, corr.shape[0] * 4))  # [25:285, 25:285]
                img = sat[0].permute(1, 2, 0).data.cpu().numpy()[
                    (512 - prob_map.shape[0]) // 2: (-512 + prob_map.shape[0]) // 2,
                    (512 - prob_map.shape[0]) // 2: (-512 + prob_map.shape[0]) // 2, :]
                cmap_name = 'rainbow' # 使用反转的 coolwarm

                overlay = show_cam_on_image(img, prob_map, False, cmap_name)

                fig, ax = plt.subplots()
                shw = ax.imshow(overlay)
                A = overlay.shape[0]
                # init = ax.scatter(A / 2, A / 2, color='r', linewidth=1, edgecolor="w", s=160, zorder=2)
                
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

                # pred = ax.scatter(pred_u + A / 2, pred_v + A / 2, linewidth=1, edgecolor="w", color='r',
                #                 s=240, zorder=2)

                # # import pdb; pdb.set_trace()
                # gt = ax.scatter(gt_u1[g_idx] + A / 2, gt_v1[g_idx] + A / 2, color='g', linewidth=1,
                #                 edgecolor="w", marker="*",
                #                 s=400,
                #                 zorder=2)


                # weakly = ax.scatter(
                #     int(gt_u1[0]) + A / 2 + random.random() * 120, 
                #     int(gt_v1[0]) + A / 2 + random.random() * 120, 
                #     color='yellow', 
                #     s=200, 
                #     edgecolor='w', 
                #     marker='s', 
                #     alpha=0.8
                #     )
                
                # forward = ax.scatter(
                #     int(gt_u1[0]) + A / 2 + random.random() * 90, 
                #     int(gt_v1[0]) + A / 2 + random.random() * 90, 
                #     color='blue', 
                #     s=200, 
                #     edgecolor='w', 
                #     marker='d', 
                #     alpha=0.8
                #     )

                pred = ax.scatter(
                    int(pred_u) + A / 2, 
                    int(pred_v) + A / 2, 
                    color='green', 
                    s=200, 
                    edgecolor='w', 
                    marker='o', 
                    alpha=0.8
                    )
                gt = ax.scatter(
                    int(gt_u1[0]) + A / 2, 
                    int(gt_v1[0]) + A / 2, 
                    color='red', 
                    s=400, 
                    edgecolor='w', 
                    marker='$⚑$', 
                    alpha=0.8
                    )
                # ax.legend([pred, gt], ['Pred', 'GT'], markerscale=1.2, frameon=False, fontsize=16,
                #         edgecolor="w", labelcolor='w', shadow=True, facecolor='b', loc='upper right')
                
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

                # ax.legend(
                #     (weakly, forward, pred, gt),
                #     ('WeaklyG2S', 'Direct porjection', 'Ours', 'GT'),
                #     markerscale=1.2,
                #     frameon=False,
                #     fontsize=16,
                #     edgecolor="black",
                #     labelcolor='black',
                #     shadow=True,
                #     facecolor='w',
                #     loc='upper center',  # 让 legend 在上方居中
                #     bbox_to_anchor=(0.5, 1.25),  # 设置 legend 位置，1.05 表示在图像上方
                #     ncol=2
                # )
                
                # plt.savefig(
                #     os.path.join(visualize_dir,
                #                 'pos_' + str(i * args.batch_size + 0) + '_' + str(
                #                     0) + '.png'),
                #     transparent=True, dpi=150, bbox_inches='tight', pad_inches=0.5)
                # plt.close()

                plt.savefig(
                    'seq/corr.png',
                    transparent=True, dpi=150, bbox_inches='tight', pad_inches=0.5)
                plt.close()

                test_img = to_pil_img(grd[0])
                test_img.save(os.path.join(visualize_dir, 'grd.png'))
                test_img = to_pil_img(sat[0])
                test_img.save(os.path.join(visualize_dir, 'sat.png'))   
                            # else:
                            #     ax.legend([pred], ['Pred'], markerscale=1.2, frameon=False,
                            #             fontsize=16,
                            #             edgecolor="w", labelcolor='w', shadow=True, facecolor='b', loc='upper right')

                            #     plt.savefig(
                            #         os.path.join(visualize_dir,
                            #                     'neg_' + str(Loop * args.batch_size + g_idx) + '_' + str(
                            #                         s_idx) + '.png'),
                            #         transparent=True, dpi=150, bbox_inches='tight', pad_inches=-0.1)

                            #     plt.close()
                print('done')


            pred_u = pred_u * meter_per_pixel
            pred_v = pred_v * meter_per_pixel

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_shift_u = gt_shift_u * meter_per_pixel * 512 / 4
            gt_shift_v = gt_shift_v * meter_per_pixel * 512 / 4

            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())

            if i % 20 == 0:
                print(i)

    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)


    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'val_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))

    line = 'args.stage: ' + str(args.stage) + 'stage: ' + str(stage) + '\n'
    print(line)
    f.write(line)

    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)

    line = 'Distance average: (init, pred)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')


    print('====================================')
    f.write('====================================\n')
    f.close()

    result = np.mean(distance)

    net.train()

    ### save the best params
    if args.stage > 0 or (args.stage == -1 and stage == 2):
        if (result < best):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(net.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    print('Finished Val')
    return result


def train(net, args, save_path):
    bestResult = 0.0

    if args.Supervision == 'Weakly':
        # params = list(net.SatFeatureNet.parameters()) + list(net.GrdFeatureNet.parameters())
        if args.share:
            params = net.dpt_sat.parameters()
        else:
            params = list(net.dpt_grd.parameters()) + list(net.dpt_sat.parameters())
        optimizer = optim.AdamW(params, lr= 3.5e-4, weight_decay=5e-3, eps=1e-8)
        # TODO: change strategy to linear
        scale = float(args.batch_size / 8)
        scheduler = OneCycleLR(optimizer, 
                                max_lr=args.lr,  # 3.5e-4
                                steps_per_epoch=int(5260 / scale), 
                                epochs=args.epochs, # 5
                                anneal_strategy='cos',
                                pct_start=0.01,
                                cycle_momentum=False,
                                )
    
    else:
        params = net.gaussian_encoder.parameters()
        optimizer = optim.Adam(params, lr=1e-4)

    time_start = time.time()    
    for epoch in range(args.resume, args.epochs):
        net.train()

        # params = list(net.GrdFeatureNet.parameters()) + list(net.SatFeatureNet.parameters())
        isTrain = True
        trainloader, valloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                                 train=isTrain, weak_supervise=args.Supervision=='Weakly', amount=args.amount)

        # val(valloader, net, args, save_path, epoch, best=bestResult, stage=args.stage)

        # return KeyError('quit')
        print('batch_size:', args.batch_size, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):

            if args.Supervision == 'Weakly':
                grd, sat, depth_imgs, grd_ori, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

                sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict = \
                    net(sat, grd, depth_imgs, grd_ori, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage=args.stage, loop=Loop, save_dir=save_path)

                corr_maps = batch_wise_cross_corr(sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, args, sat_uncer_dict)

                if args.visualize:

                    visualize_dir = os.path.join(save_path, 'visualization')
                    if not os.path.exists(visualize_dir):
                        os.makedirs(visualize_dir)

                    level = max([int(item) for item in args.level.split('_')])
                    corr = corr_maps[level]
                    corr_H, corr_W = corr.shape[2:]

                    gt_u1 = gt_shift_u.data.cpu().numpy() * 512/4
                    gt_v1 = gt_shift_v.data.cpu().numpy() * 512/4

                    for g_idx in range(corr.shape[0]):
                        for s_idx in range(corr.shape[1]):
                            if g_idx == s_idx:
                                visualize_dir = os.path.join(save_path, 'visualization', f"{Loop}_{g_idx}")
                                if not os.path.exists(visualize_dir):
                                    os.makedirs(visualize_dir)
                                
                                max_index = torch.argmin(corr[g_idx, s_idx].reshape(-1)).data.cpu().numpy()

                                pred_u = (max_index % corr_W - corr_W / 2) * np.power(2, 3 - level)
                                pred_v = (max_index // corr_W - corr_H / 2) * np.power(2, 3 - level)

                                prob_map = cv2.resize(corr[g_idx, s_idx].data.cpu().numpy(),
                                                    (corr.shape[3] * 4, corr.shape[2] * 4))  # [25:285, 25:285]
                                img = sat[s_idx].permute(1, 2, 0).data.cpu().numpy()[
                                    (512 - prob_map.shape[0]) // 2: (-512 + prob_map.shape[0]) // 2,
                                    (512 - prob_map.shape[0]) // 2: (-512 + prob_map.shape[0]) // 2, :]

                                overlay = show_cam_on_image(img, prob_map, False, cv2.COLORMAP_HSV)

                                fig, ax = plt.subplots()
                                shw = ax.imshow(overlay)
                                A = overlay.shape[0]
                                # init = ax.scatter(A / 2, A / 2, color='r', linewidth=1, edgecolor="w", s=160, zorder=2)
                                
                                ax.axis('off')

                                # pred = ax.scatter(pred_u + A / 2, pred_v + A / 2, linewidth=1, edgecolor="w", color='r',
                                #                 s=240, zorder=2)

                                # # import pdb; pdb.set_trace()
                                # gt = ax.scatter(gt_u1[g_idx] + A / 2, gt_v1[g_idx] + A / 2, color='g', linewidth=1,
                                #                 edgecolor="w", marker="*",
                                #                 s=400,
                                #                 zorder=2)
                                pred = ax.scatter(
                                    int(pred_u) + A / 2, 
                                    int(pred_v) + A / 2, 
                                    color='green', 
                                    s=200, 
                                    edgecolor='w', 
                                    marker='o', 
                                    alpha=0.8
                                    )
                                gt = ax.scatter(
                                    int(gt_u1[g_idx]) + A / 2 + 10, 
                                    int(gt_v1[g_idx]) + A / 2 - 15, 
                                    color='red', 
                                    s=400, 
                                    edgecolor='w', 
                                    marker='$⚑$', 
                                    alpha=0.8
                                    )
                                # ax.legend([pred, gt], ['Pred', 'GT'], markerscale=1.2, frameon=False, fontsize=16,
                                #         edgecolor="w", labelcolor='w', shadow=True, facecolor='b', loc='upper right')
                                
                                ax.legend(
                                    (pred, gt),
                                    ('Ours', 'GT'),
                                    markerscale=1.2,
                                    frameon=False,
                                    fontsize=12,
                                    edgecolor="black",
                                    labelcolor='black',
                                    shadow=True,
                                    facecolor='w',
                                    loc='upper center',  # 让 legend 在上方居中
                                    bbox_to_anchor=(0.5, 1.12),  # 设置 legend 位置，1.05 表示在图像上方
                                    ncol=3
                                )
                                
                                plt.savefig(
                                    os.path.join(visualize_dir,
                                                'pos_' + str(Loop * args.batch_size + g_idx) + '_' + str(
                                                    s_idx) + '.png'),
                                    transparent=True, dpi=150, bbox_inches='tight', pad_inches=-0.1)
                                plt.close()

                                test_img = to_pil_img(grd[g_idx])
                                test_img.save(os.path.join(visualize_dir, 'grd.png'))
                                test_img = to_pil_img(sat[s_idx])
                                test_img.save(os.path.join(visualize_dir, 'sat.png'))   
                                # else:
                                #     ax.legend([pred], ['Pred'], markerscale=1.2, frameon=False,
                                #             fontsize=16,
                                #             edgecolor="w", labelcolor='w', shadow=True, facecolor='b', loc='upper right')

                                #     plt.savefig(
                                #         os.path.join(visualize_dir,
                                #                     'neg_' + str(Loop * args.batch_size + g_idx) + '_' + str(
                                #                         s_idx) + '.png'),
                                #         transparent=True, dpi=150, bbox_inches='tight', pad_inches=-0.1)

                                #     plt.close()
                        print('done')

                corr_loss, GPS_loss = Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v,
                                                                         args,
                                                                         meter_per_pixel,
                                                                         args.GPS_error)

                # loss = corr_loss + args.GPS_error_coe * GPS_loss
                loss = corr_loss + GPS_loss * args.GPS_error_coe
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if Loop % 10 == 9:  #
                    time_end = time.time()
                    current_lr = scheduler.get_last_lr()[0]
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                          ' GPS loss: ' + str(np.round(GPS_loss.item(), decimals=4)) +
                          f' Learning Rate: {current_lr:.9f}' +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end
            
            else:
                grd, sat, depth_imgs, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]
                loss = \
                    net(sat, grd, depth_imgs, meter_per_pixel, gt_rot, gt_shift_u, gt_shift_v, stage=args.stage, loop=Loop, save_dir=save_path)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if Loop % 10 == 9:  #
                    time_end = time.time()
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                          ' Render loss: ' + str(np.round(loss.item(), decimals=4)) +
                          ' Time: ' + str(time_end - time_start))

                    time_start = time_end

        print('Save Model ...', save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        if args.Supervision == 'Weakly':
            bestResult = val(valloader, net, args, save_path, epoch, best=bestResult, stage=args.stage)


    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')

    parser.add_argument('--rotation_range', type=float, default=0., help='degree')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    parser.add_argument('--level', type=str, default='1', help=' ')
    parser.add_argument('--channels', type=str, default='32_16_4', help=' ')
    parser.add_argument('--N_iters', type=int, default=1, help='any integer')

    parser.add_argument('--Optimizer', type=str, default='TransV1', help='LM or SGD')
    parser.add_argument('--proj', type=str, default='geo', help='geo, polar, nn, CrossAttn')
    parser.add_argument('--use_uncertainty', type=int, default=0, help='0 or 1')

    parser.add_argument('--area', type=str, default='same', help='same or cross')
    parser.add_argument('--multi_gpu', type=int, default=0, help='0 or 1')

    parser.add_argument('--ConfGrd', type=int, default=1, help='use confidence or not for grd image')
    parser.add_argument('--ConfSat', type=int, default=0, help='use confidence or not for sat image')

    parser.add_argument('--share', type=int, default=0, help='share feature extractor for grd and sat or not '
                                                             'in translation estimation')

    parser.add_argument('--GPS_error', type=int, default=5, help='')
    parser.add_argument('--GPS_error_coe', type=float, default=0., help='')

    parser.add_argument('--stage', type=int, default=3,
                        help='fix to 3, this is for dataloader')
    parser.add_argument('--task', type=str, default='2DoF',
                        help='')

    parser.add_argument('--Supervision', type=str, default='Weakly',
                        help='Weakly or Gaussian')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 1')

    parser.add_argument('--sat', type=float, default=0., help='')
    parser.add_argument('--grd', type=float, default=0., help='')
    parser.add_argument('--sat_grd', type=float, default=1., help='')
    parser.add_argument('--amount', type=float, default=1., help='')
    parser.add_argument('--name', type=str, default='pano_cuda', help='')
    parser.add_argument('--grd_res', type=int, default=40, help='')    
    parser.add_argument('--lr', type=float, default=1e-4, help='')    

    args = parser.parse_args()

    return args


def getSavePath(args):
    restore_path = '/data/qiwei/nips25/CVLnet2/ModelsVIGOR/' + str(args.task) \
                   + '/' + args.area + '_rot' + str(args.rotation_range) \
                   + '_' + str(args.proj) \
                   + '_' + str(args.lr) \
                   + '_Level' + args.level + '_Channels' + args.channels

    save_path = restore_path

    if args.ConfGrd:
        save_path = save_path + '_ConfGrd'
    if args.ConfSat:
        save_path = save_path + '_ConfSat'


    if args.GPS_error_coe > 0:
        save_path = save_path + '_GPSerror' + str(args.GPS_error) + '_Coe' + str(args.GPS_error_coe)

    if args.share:
        save_path = save_path + '_Share'

    save_path += '_' + args.Supervision
    save_path += '_' + args.name

    print('save_path:', save_path)

    return save_path, restore_path


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    save_path, restore_path = getSavePath(args)

    net = ModelVIGOR(args, device)
    net.to(device)

    if args.test:
        # net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')), strict=False)
        # current = test(net, args, save_path)
        # path = 'ModelsVIGOR/2DoF/cross_rot0.0_geo_0.0001_Level1_Channels32_16_4_ConfGrd_Weakly_vigor_1.0/model_6.pth'
        # path = '/data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/same_rot0.0_geo_0.000125_Level1_Channels32_16_4_ConfGrd_Weakly_vigor_0.3_3.0_70_1.25e-4_depth/model_13.pth'
        # path = '/data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/same_rot0.0_geo_6.5e-05_Level1_Channels32_16_4_ConfGrd_Weakly_vigor_1.0/model_13.pth'
        ### cross path
        # path = '/data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/cross_rot0.0_geo_6.5e-05_Level1_Channels32_16_4_ConfGrd_Weakly_vigor_1.0_70_6.5e-5/model_7.pth'
        # path ='/data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/cross_rot0.0_geo_0.000125_Level1_Channels32_16_4_ConfGrd_Weakly_vigor_1.25_GPS/model_14.pth'
        path = save_path + '/model_14.pth'
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
        print("Test from " + path)
        _, testloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                                 train=False, weak_supervise=args.Supervision=='Weakly', amount=args.amount)
        val(testloader, net, args, save_path, 0, stage=args.stage)

    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')),
                                strict=False)
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')
        # else:
        #     path = '/home/wangqw/video_program/CVLNet2/ModelsVIGOR/2DoF/same_rot0.0_geo_Level1_Channels32_16_4_ConfGrd_Gaussian_20face_new80/model_2.pth'
        #     net.load_state_dict(torch.load(path), strict=False)
        train(net, args, save_path)

