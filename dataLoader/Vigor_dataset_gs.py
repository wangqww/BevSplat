import random

import numpy as np
import os
from PIL import Image
import PIL
from torch.utils.data import Dataset, Subset

import torch

from torch.utils.data import DataLoader
from torchvision import transforms

num_thread_workers = 64
root = '/data/dataset/VIGOR'

class VIGORDataset(Dataset):
    def __init__(self, 
                 root, 
                 rotation_range, 
                 label_root='splits__corrected', 
                 split='same', 
                 train=True, 
                 transform=None, 
                 pos_only=True, 
                 amount=1.,
                 width=160,
                 height=80
                 ):
        self.root = root
        self.rotation_range = rotation_range
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only
        self.height = height
        self.width = width
        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]

        if self.split == 'same':
            self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.split == 'cross':
            if self.train:
                self.city_list = ['NewYork', 'Seattle']
            else:
                self.city_list = ['SanFrancisco', 'Chicago']

        self.meter_per_pixel_dict = {'NewYork': 0.113248 * 640 / 512,
                                     'Seattle': 0.100817 * 640 / 512,
                                     'SanFrancisco': 0.118141 * 640 / 512,
                                     'Chicago': 0.111262 * 640 / 512}

        # load sat list
        self.sat_list = []
        self.sat_index_dict = {}

        idx = 0
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', sat_list_fname, idx)
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)
        print('Sat loaded, data size:{}'.format(self.sat_data_size))

        # load grd list
        self.grd_list = []
        self.ori_grd_list = []
        self.depth_list = []
        self.label = []
        self.sat_cover_dict = {}
        self.delta = []
        idx = 0
        for city in self.city_list:
            # load grd panorama list
            if self.split == 'same':
                if self.train:
                    label_fname = os.path.join(self.root, self.label_root, city, 'same_area_balanced_train__corrected.txt')
                else:
                    label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test__corrected.txt')
            elif self.split == 'cross':
                label_fname = os.path.join(self.root, self.label_root, city, 'pano_label_balanced__corrected.txt')

            with open(label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.sat_index_dict[data[i]])
                    label = np.array(label).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.grd_list.append(os.path.join(self.root, city, 'pano_mask_sky', data[0]))
                    self.ori_grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.depth_list.append(os.path.join(self.root, city, f'UniK3D_{split}_metric', data[0].replace('.jpg', '_depth.npy')))
                    # self.depth_list.append(os.path.join(self.root, city, 'depth_anywhere_same', data[0].replace('.jpg', '_depth.png')))
                    # self.depth_list.append(os.path.join(self.root, city, 'pers_imgs_160_new', data[0].replace('.jpg', '_pers.pt')))
                    # self.grd_params.append(os.path.join(self.root, city, 'pers_imgs', data[0].replace('.jpg', '_pers.pt')))
                    self.label.append(label)
                    self.delta.append(delta)
                    if not label[0] in self.sat_cover_dict:
                        self.sat_cover_dict[label[0]] = [idx]
                    else:
                        self.sat_cover_dict[label[0]].append(idx)
                    idx += 1

            print('InputData::__init__: load ', label_fname, idx)

        self.data_size = int(len(self.grd_list) * amount)
        self.grd_list = self.grd_list[: self.data_size]
        self.ori_grd_list = self.ori_grd_list[: self.data_size]
        self.label = self.label[: self.data_size]
        self.delta = self.delta[: self.data_size]
        print('Grd loaded, data size:{}'.format(self.data_size))
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)
        self.direction = get_panorama_ray_directions(self.height, self.width)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        # full ground panorama
        try:
            grd = PIL.Image.open(os.path.join(self.grd_list[idx]))
            grd = grd.convert('RGB')
        except:
            print('unreadable image')
            grd = PIL.Image.new('RGB', (320, 640))  # if the image is unreadable, use a blank image
        grd = self.grdimage_transform(grd)

        try:
            grd_ori = PIL.Image.open(os.path.join(self.ori_grd_list[idx]))
            grd_ori = grd_ori.convert('RGB')
        except:
            print('unreadable image')
            grd_ori = PIL.Image.new('RGB', (320, 640))  # if the image is unreadable, use a blank image
        grd_ori = self.grdimage_transform(grd_ori)

        # try:
        #     depth = PIL.Image.open(os.path.join(self.depth_list[idx]))
        #     depth = depth.convert('L')
        # except:
        #     print('unreadable image')
        #     depth = PIL.Image.new('L', (320, 640))

        # depth_img = self.grdimage_transform(depth)
        
        depth_img = np.load(self.depth_list[idx])
        depth_img = torch.tensor(depth_img, dtype=torch.float32)
        # depth_img = torch.load(self.depth_list[idx], map_location='cpu')
        # depth_img = depth_img['depth_imgs']
        
        # generate a random rotation
        rotation = np.random.uniform(low=-1.0, high=1.0)  #
        rotation_angle = rotation * self.rotation_range
        grd = torch.roll(grd, (torch.round(torch.as_tensor(rotation_angle / 180) * grd.size()[2] / 2).int()).item(),
                         dims=2)

        # satellite
        if self.pos_only:  # load positives only
            pos_index = 0
            sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
            [row_offset, col_offset] = self.delta[idx, pos_index]  # delta = [delta_lat, delta_lon]
        else:  # load positives and semi-positives
            col_offset = 320
            row_offset = 320
            while (np.abs(col_offset) >= 320 or np.abs(
                    row_offset) >= 320):  # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0, 3)
                sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
                [row_offset, col_offset] = self.delta[idx, pos_index]  # delta = [delta_lat, delta_lon]

        sat = sat.convert('RGB')
        width_raw, height_raw = sat.size

        sat = self.satimage_transform(sat)
        _, height, width = sat.size()
        row_offset = np.round(row_offset / height_raw * height)
        col_offset = np.round(col_offset / width_raw * width)

        # groundtruth location on the aerial image
        gt_shift_y = row_offset / height * 4  # -L/4 ~ L/4  -1 ~ 1
        gt_shift_x = -col_offset / width * 4  #

        if 'NewYork' in self.grd_list[idx]:
            city = 'NewYork'
        elif 'Seattle' in self.grd_list[idx]:
            city = 'Seattle'
        elif 'SanFrancisco' in self.grd_list[idx]:
            city = 'SanFrancisco'
        elif 'Chicago' in self.grd_list[idx]:
            city = 'Chicago'

        return grd, sat, depth_img, grd_ori, \
            torch.tensor(gt_shift_x, dtype=torch.float32), \
            torch.tensor(gt_shift_y, dtype=torch.float32), \
            torch.tensor(rotation, dtype=torch.float32), \
            torch.tensor(self.meter_per_pixel_dict[city], dtype=torch.float32)

# ---------------------------------------------------------------------------------



def load_vigor_data(batch_size, area="same", rotation_range=0, train=True, weak_supervise=True, amount=1.):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    transform_grd = transforms.Compose([
        transforms.Resize([320, 640]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    transform_sat = transforms.Compose([
        # resize
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    vigor = VIGORDataset(root, rotation_range, split=area, train=train, transform=(transform_grd, transform_sat),
                         amount=amount)
    
    if train:

        index_list = np.arange(vigor.__len__())
        # np.random.shuffle(index_list)
        train_indices = index_list[0: int(len(index_list) * 0.8)]
        val_indices = index_list[int(len(index_list) * 0.8):]
        training_set = Subset(vigor, train_indices)
        # training_set = Subset(vigor, range(20))
        val_set = Subset(vigor, val_indices)

        train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_thread_workers)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_thread_workers)
        
        return train_dataloader, val_dataloader

    else:
        index_list = np.arange(vigor.__len__())
        val_indices = index_list[0: int(len(index_list))]
        val_set = Subset(vigor, val_indices)
        test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_thread_workers)

        return None, test_dataloader

def get_panorama_ray_directions(
    H: int,
    W: int,
):  
    # 创建 theta 和 phi 为 1D 张量
    theta = torch.linspace(0, 2 * torch.pi, W)  # 方位角 [0, 2π]
    phi = torch.linspace(0, torch.pi, H)       # 仰角 [0, π]
    
    # 生成网格，调整 indexing='ij' 确保符合 PyTorch 约定
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')

    # 计算 OpenCV 形式的 X, Y, Z 坐标
    x = -torch.sin(phi) * torch.sin(theta)   # OpenCV X: 右
    y = -torch.cos(phi)                     # OpenCV Y: 下
    z = -torch.sin(phi) * torch.cos(theta)  # OpenCV Z: 前

    # 将 x, y, z 堆叠在一起，并调整维度 (height, width, 3)
    directions = torch.stack((x, y, z), dim=-1)  # (B, H, W, 3)
    
    return directions