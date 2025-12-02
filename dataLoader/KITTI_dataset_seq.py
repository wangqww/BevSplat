import random

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import pandas as pd
import boost_utils
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

root_dir = '/data/dataset/KITTI' # '../../data/Kitti' # '../Data' #'..\\Data' #
# root_dir = '/media/yujiao/6TB/dataset/Kitti1'

satmap_dir = 'satmap'
grdimage_dir = 'depth_data'
grd_depth_dir = 'image_02/grd_depth'  # 'image_02\\data' #
left_color_camera_dir = 'image_02/grd_no_sky'  # 'image_02\\data' #
left_color_camera_dir_ori = 'image_02/data'  # 'image_02\\data' #
right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #
# depth_dir = 'depth/data_depth_annotated/train/'

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 32

# train_file = './dataLoader/train_files.txt'
train_file = './dataLoader/train_files.txt'
test1_file = './dataLoader/test1_files.txt'
test2_file = './dataLoader/test2_files.txt'


# def depth_read(filename):
#     # loads depth map D from png file
#     # and returns it as a numpy array,
#     # for details see readme.txt

#     depth_png = np.array(Image.open(filename), dtype=int)
#     # make sure we have a proper 16bit depth map here.. not 8bit!
#     assert(np.max(depth_png) > 255)

#     depth = depth_png.astype(np.float) / 256.
#     depth[depth_png == 0] = -1.
#     return depth


class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10, sequence=4):
        self.root = root
        self.sequence = sequence
        self.meter_per_pixel = boost_utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'depth_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        # np.random.seed(2022)
        # num = len(file_name)//3
        # random.shuffle(file_name)
        # self.file_name = [file[:-1] for file in file_name[:num]]
        self.file_name = [file[:-1] for file in file_name]
        # self.file_name = []
        # count = 0
        # for file in file_name:
        #     left_depth_name = os.path.join(self.root, depth_dir, file.split('/')[1],
        #                                    'proj_depth/groundtruth/image_02', os.path.basename(file.strip()))
        #     if os.path.exists(left_depth_name):
        #         self.file_name.append(file.strip())
        #     else:
        #         count += 1
        #
        # print('number of files whose depth unavailable: ', count)


    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read file names within one sequence =====================
        sequence_list = []
        if self.sequence > 1:
            # need get sequence count files
            sequence_count = self.sequence

            # get sequence count files in drive_dir in before, if not enough, get after
            sequence_list.append(file_name)
            tar_image_no = int(image_no.split('.')[0])
            while len(sequence_list) < sequence_count:
                tar_image_no = tar_image_no - self.skip_in_seq - 1

                # create name of
                tar_img_no = '%010d' % (tar_image_no) + '.png'
                tar_file_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, right_color_camera_dir, tar_img_no)
                if os.path.exists(tar_file_name):
                    sequence_list.append(drive_dir + tar_img_no)
                else:
                    print('error, no enough sequence images in drive_dir:', drive_dir, len(sequence_list))
                    break
        else:
            sequence_list.append(file_name)

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        loc_left_array = torch.tensor([])
        heading_array = torch.tensor([])
        real_gps = torch.tensor([])
        grd_left_depths = torch.tensor([])
        grd_left_imgs_ori = torch.tensor([])

        for i in range(len(sequence_list)):
            file_name = sequence_list[i]
            image_no = file_name[38:]

            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                        image_no.lower().replace('.png', '.txt'))
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')
                
                # get heading
                heading = float(content[5])

                # get location
                # utm_x, utm_y = utils.gps2utm(float(content[0]), float(content[1]), lat0)  # location of the GPS device
                utm_x, utm_y = boost_utils.gps2utm(float(content[0]), float(content[1]))
                delta_left_x, delta_left_y = boost_utils.get_camera_gps_shift_left(
                    heading)  # delta x and delta y between the GPS and the left camera device
                
                left_x = utm_x + delta_left_x
                left_y = utm_y + delta_left_y
                
                loc_left = torch.from_numpy(np.asarray([left_x, left_y]))
                heading = torch.from_numpy(np.asarray(heading))
                real_gps = torch.cat([real_gps, torch.from_numpy(np.asarray([float(content[0]), float(content[1])])).unsqueeze(0)])
                
                left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                            image_no.lower())
                
                left_img_name_ori = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir_ori,
                                            image_no.lower())
                                
                with Image.open(left_img_name, 'r') as GrdImg:
                    grd_img_left = GrdImg.convert('RGB')
                    if self.grdimage_transform is not None:
                        grd_img_left = self.grdimage_transform(grd_img_left)

                with Image.open(left_img_name_ori, 'r') as GrdImg:
                    grd_img_left_ori = GrdImg.convert('RGB')
                    if self.grdimage_transform is not None:
                        grd_img_left_ori = self.grdimage_transform(grd_img_left_ori)

                grd_depth = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, grd_depth_dir,
                                image_no.lower().replace('.png', '_grd_depth.pt'))

                grd_depth_left = torch.load(grd_depth, map_location=torch.device('cpu'), weights_only=True)
                # left_depth = F.interpolate(left_depth[None, None, :, :], (GrdImg_H, GrdImg_W))
                # left_depth = left_depth[0, 0]

                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
                grd_left_imgs_ori = torch.cat([grd_left_imgs_ori, grd_img_left_ori.unsqueeze(0)], dim=0)
                grd_left_depths = torch.cat([grd_left_depths, grd_depth_left.unsqueeze(0)], dim=0)
                
                loc_left_array = torch.cat([loc_left_array, loc_left.unsqueeze(0)], dim=0)
                heading_array = torch.cat([heading_array, heading.unsqueeze(0)], dim=0)
        
        locations = []
        prev_img_data = None
        loc_left_array = torch.flip(loc_left_array, [0])
        heading_array_flip = torch.flip(heading_array, [0])
        for frame in range(len(loc_left_array)):
            if prev_img_data is not None:
                x = loc_left_array[frame][0] - prev_img_data[0]
                y = loc_left_array[frame][1] - prev_img_data[1]
                gps_distance = torch.sqrt(torch.pow(x, 2)+torch.pow(y,2))
                yaw_change = heading_array_flip[frame] - prev_img_data[2]
                for i in range(len(locations)):
                    x0, y0 = locations[i]
                    x1 = x0 * torch.cos(yaw_change) + y0 * torch.sin(yaw_change) - gps_distance
                    y1 = -x0 * torch.sin(yaw_change) + y0 * torch.cos(yaw_change)
                    locations[i] = torch.tensor([x1,y1], dtype=torch.float32)
            locations += [torch.tensor([0,0], dtype=torch.float32)]
            prev_img_data = [loc_left_array[frame][0], loc_left_array[frame][1], heading_array[frame]]
        locations = torch.stack(locations)
        locations = torch.flip(locations, [0])
        
        heading_shift_left = (heading_array - heading_array[0])
        
        # loc_shift_left = (loc_left_array - loc_left_array[0:1, :])

        # 旋转矩阵
        # R = torch.tensor([[torch.cos(heading_array[0] / torch.pi * 180), -torch.sin(heading_array[0] / torch.pi * 180)],
        #               [torch.sin(heading_array[0] / torch.pi * 180), torch.cos(heading_array[0] / torch.pi * 180)]])
        # loc_shift_left = torch.matmul(loc_shift_left, R.T)
        
        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        # randomly generate roation
        theta = np.random.uniform(-1, 1)
        
        gt_shift_xs = -locations[:,0] / self.shift_range_pixels_lon - gt_shift_x
        gt_shift_ys = -locations[:,1] / self.shift_range_pixels_lat - gt_shift_y
        thetas = heading_shift_left / torch.pi * 180 / max(self.rotation_range, 1e-6) + theta
        
        sat_rot = sat_map.rotate(-heading_array[0] / torch.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, boost_utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, boost_utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)

        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(theta * self.rotation_range)

        sat_map =TF.center_crop(sat_rand_shift_rand_rot, boost_utils.SatMap_process_sidelength)
        # sat_map = np.array(sat_map, dtype=np.float32)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)
        
        # gt_corr_x, gt_corr_y = self.generate_correlation_GTXY(gt_shift_x, gt_shift_y, theta)

        return sat_map, left_camera_k, grd_left_imgs, grd_left_imgs_ori,  grd_left_depths, gt_shift_xs.float(), gt_shift_ys.float(), thetas.float(), locations.float(), heading_shift_left.float(), real_gps.float(), file_name


    # def generate_correlation_GTXY(self, gt_shift_x, gt_shift_y, gt_heading):
        
    #     cos = np.cos(gt_heading * self.rotation_range / 180 * np.pi)
    #     sin = np.sin(gt_heading * self.rotation_range / 180 * np.pi)
        
    #     gt_corr_x = - gt_shift_x * cos + gt_shift_y * sin
    #     gt_corr_y = gt_shift_x * sin + gt_shift_y * cos
        
    #     return gt_corr_x, gt_corr_y
        
        
        



class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10, sequence = 4):
        self.root = root
        self.sequence = sequence
        self.meter_per_pixel = boost_utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'depth_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        # np.random.seed(2022)
        # num = len(file_name)//3
        # random.shuffle(file_name)
        # self.file_name = [file[:-1] for file in file_name[:num]]
        self.file_name = [file[:-1] for file in file_name]
        # self.file_name = []
        # count = 0
        # for line in file_name:
        #     file = line.split(' ')[0]
        #     left_depth_name = os.path.join(self.root, depth_dir, file.split('/')[1],
        #                                    'proj_depth/groundtruth/image_02', os.path.basename(file.strip()))
        #     if os.path.exists(left_depth_name):
        #         self.file_name.append(line.strip())
        #     else:
        #         count += 1
        #
        # print('number of files whose depth unavailable: ', count)


    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read file names within one sequence =====================
        sequence_list = []
        if self.sequence > 1:
            # need get sequence count files
            sequence_count = self.sequence

            # get sequence count files in drive_dir in before, if not enough, get after
            sequence_list.append(file_name)
            tar_image_no = int(image_no.split('.')[0])
            while len(sequence_list) < sequence_count:
                tar_image_no = tar_image_no - self.skip_in_seq - 1

                # create name of
                tar_img_no = '%010d' % (tar_image_no) + '.png'
                tar_file_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, right_color_camera_dir, tar_img_no)
                if os.path.exists(tar_file_name):
                    sequence_list.append(drive_dir + tar_img_no)
                else:
                    print('error, no enough sequence images in drive_dir:', drive_dir, len(sequence_list))
                    break
        else:
            sequence_list.append(file_name)
        
        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        gt_shift_xs = torch.tensor([])
        gt_shift_ys = torch.tensor([])
        thetas = torch.tensor([])
        loc_left_array = torch.tensor([])
        heading_array = torch.tensor([])
        real_gps = torch.tensor([])
        # image_no = file_name[38:]
        grd_left_imgs_ori = torch.tensor([])
        grd_left_depths = torch.tensor([])
        
        for i in range(len(sequence_list)):
            file_name = sequence_list[i]
            image_no = file_name[38:]
            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                        image_no.lower().replace('.png', '.txt'))
            
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')

                # get heading
                heading = float(content[5])
                
                real_gps = torch.cat([real_gps, torch.from_numpy(np.asarray([float(content[0]), float(content[1])])).unsqueeze(0)])
                
                # get location
                # utm_x, utm_y = utils.gps2utm(float(content[0]), float(content[1]), lat0)  # location of the GPS device
                utm_x, utm_y = boost_utils.gps2utm(float(content[0]), float(content[1]))
                delta_left_x, delta_left_y = boost_utils.get_camera_gps_shift_left(
                    heading)  # delta x and delta y between the GPS and the left camera device
                
                left_x = utm_x + delta_left_x
                left_y = utm_y + delta_left_y
                
                loc_left = torch.from_numpy(np.asarray([left_x, left_y]))
                heading = torch.from_numpy(np.asarray(heading))
                
                left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                            image_no.lower())
                
                left_img_name_ori = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir_ori,
                                            image_no.lower())

                with Image.open(left_img_name, 'r') as GrdImg:
                    grd_img_left = GrdImg.convert('RGB')
                    if self.grdimage_transform is not None:
                        grd_img_left = self.grdimage_transform(grd_img_left)

                with Image.open(left_img_name_ori, 'r') as GrdImg:
                    grd_img_left_ori = GrdImg.convert('RGB')
                    if self.grdimage_transform is not None:
                        grd_img_left_ori = self.grdimage_transform(grd_img_left_ori)

                grd_depth = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, grd_depth_dir,
                                image_no.lower().replace('.png', '_grd_depth.pt'))
                
                grd_depth_left = torch.load(grd_depth, map_location=torch.device('cpu'), weights_only=True)


                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
                grd_left_imgs_ori = torch.cat([grd_left_imgs_ori, grd_img_left_ori.unsqueeze(0)], dim=0)                
                grd_left_depths = torch.cat([grd_left_depths, grd_depth_left.unsqueeze(0)], dim=0)

                loc_left_array = torch.cat([loc_left_array, loc_left.unsqueeze(0)], dim=0)
                heading_array = torch.cat([heading_array, heading.unsqueeze(0)], dim=0)
        
        
        locations = []
        prev_img_data = None
        loc_left_array_flip = torch.flip(loc_left_array, [0])
        heading_array_flip = torch.flip(heading_array, [0])
        for frame in range(len(loc_left_array_flip)):
            if prev_img_data is not None:
                x = loc_left_array_flip[frame][0] - prev_img_data[0]
                y = loc_left_array_flip[frame][1] - prev_img_data[1]
                gps_distance = torch.sqrt(torch.pow(x, 2)+torch.pow(y,2))
                yaw_change = heading_array_flip[frame] - prev_img_data[2]
                for i in range(len(locations)):
                    x0, y0 = locations[i]
                    x1 = x0 * torch.cos(yaw_change) + y0 * torch.sin(yaw_change) - gps_distance
                    y1 = -x0 * torch.sin(yaw_change) + y0 * torch.cos(yaw_change)
                    locations[i] = torch.tensor([x1,y1], dtype=torch.float32)
            locations += [torch.tensor([0,0], dtype=torch.float32)]
            prev_img_data = [loc_left_array_flip[frame][0], loc_left_array_flip[frame][1], heading_array_flip[frame]]
        locations = torch.stack(locations)
        locations = torch.flip(locations, [0])
        
        # loc_shift_left = loc_left_array - loc_left_array[0:1, :]
        
        # # 旋转矩阵
        # R = torch.tensor([[torch.cos(heading_array[0] / torch.pi * 180), -torch.sin(heading_array[0] / torch.pi * 180)],
        #               [torch.sin(heading_array[0] / torch.pi * 180), torch.cos(heading_array[0] / torch.pi * 180)]])
        # loc_shift_left = torch.matmul(loc_shift_left, R.T)
            
        heading_shift_left = (heading_array - heading_array[0])
        
        sat_rot = sat_map.rotate(-heading_array[0] / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, boost_utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, boost_utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        # randomly generate shift
        # gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        # gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction
        
        gt_shift_xs = -locations[:,0] / self.shift_range_pixels_lon - gt_shift_x
        gt_shift_ys = -locations[:,1] / self.shift_range_pixels_lat - gt_shift_y
        thetas = heading_shift_left / np.pi * 180 / max(self.rotation_range, 1e-6) + float(theta)
        
        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)

        # randomly generate roation
        # theta = np.random.uniform(-1, 1)
        theta = float(theta)
        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(theta * self.rotation_range)

        sat_map = TF.center_crop(sat_rand_shift_rand_rot, boost_utils.SatMap_process_sidelength)
        # sat_map = np.array(sat_map, dtype=np.float32)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # gt_corr_x, gt_corr_y = self.generate_correlation_GTXY(gt_shift_x, gt_shift_y, theta)
                
        return sat_map, left_camera_k, grd_left_imgs, grd_left_imgs_ori, grd_left_depths, gt_shift_xs.float(), gt_shift_ys.float(), thetas.float(), locations.float(), heading_shift_left.float(), real_gps, file_name
    
    # def generate_correlation_GTXY(self, gt_shift_x, gt_shift_y, gt_heading):
        
    #     cos = np.cos(gt_heading * self.rotation_range / 180 * np.pi)
    #     sin = np.sin(gt_heading * self.rotation_range / 180 * np.pi)
        
    #     gt_corr_x = - gt_shift_x * cos + gt_shift_y * sin
    #     gt_corr_y = gt_shift_x * sin + gt_shift_y * cos
        
    #     return gt_corr_x, gt_corr_y
        


def load_train_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10, sequence=4):
    SatMap_process_sidelength = boost_utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    train_set = SatGrdDataset(root=root_dir, sequence=sequence,file=train_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return train_loader


def load_test1_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10, sequence=4):
    SatMap_process_sidelength = boost_utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    test1_set = SatGrdDatasetTest(root=root_dir, file=test1_file,
                            transform=(satmap_transform, grdimage_transform),
                            shift_range_lat=shift_range_lat,
                            shift_range_lon=shift_range_lon,
                            rotation_range=rotation_range,
                            sequence=sequence)

    test1_loader = DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)
    return test1_loader


def load_test2_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10, sequence=4):
    SatMap_process_sidelength = boost_utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    test2_set = SatGrdDatasetTest(root=root_dir, file=test2_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range,
                              sequence=sequence)

    test2_loader = DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return test2_loader







