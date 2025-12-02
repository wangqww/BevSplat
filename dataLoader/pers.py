import numpy as np
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from scipy.spatial.transform import Rotation
from trimesh.creation import icosphere as IcoSphere
import multiprocessing

import torch


# Helper functions
def _verts_to_dirs(pt_a, pt_b, pt_c, gen_res, ratio):
    # make pt_a the sole point
    def same_z(a, b):
        return np.abs(a[2] - b[2]) < 1e-4

    assert same_z(pt_a, pt_b) or same_z(pt_b, pt_c) or same_z(pt_a, pt_c)

    if same_z(pt_a, pt_b):
        pt_a, pt_c = pt_c, pt_a
    elif same_z(pt_a, pt_c):
        pt_a, pt_b = pt_b, pt_a

    assert same_z(pt_b, pt_c)

    if np.cross(pt_c, pt_b)[2] < 0.:
        pt_b, pt_c = pt_c, pt_b

    pt_a = torch.from_numpy(pt_a)
    pt_b = torch.from_numpy(pt_b)
    pt_c = torch.from_numpy(pt_c)

    pt_m = (pt_b + pt_c) * .5
    down_vec = pt_a - pt_m
    if down_vec[2] > 0.:
        down_vec = -down_vec

    pt_center = (pt_a + pt_b + pt_c) / 3.
    right_vec = pt_c - pt_b

    right_len = torch.linalg.norm(right_vec, 2, -1).item()
    down_len = torch.linalg.norm(down_vec, 2, -1).item()
    half_len = torch.linalg.norm(pt_center - pt_b, 2, -1).item() * ratio
    right_vec = right_vec / right_len * half_len
    down_vec = down_vec / down_len * half_len
    pt_base = pt_center - right_vec - down_vec
    right_vec *= 2
    down_vec *= 2

    ii, jj = torch.meshgrid(torch.linspace(.5 / gen_res, 1. - .5 / gen_res, gen_res),
                            torch.linspace(.5 / gen_res, 1. - .5 / gen_res, gen_res),
                            indexing='ij')
    to_vec = pt_base + right_vec * .5 + down_vec * .5

    dirs = pt_base[None, None, :] + \
           down_vec[None, None, :] * ii[:, :, None] + \
           right_vec[None, None, :] * jj[:, :, None]

    pers_ratios = torch.linalg.norm(dirs, 2, -1, True) / torch.linalg.norm(to_vec, 2, -1, True)[None, None]

    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    return dirs, pers_ratios, to_vec, down_vec * .5, right_vec * .5

def direction_to_pano_coord(dirs):
    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    beta = torch.arcsin(dirs[..., 2])
    xy = dirs[..., :2] / torch.cos(beta)[..., None]
    alpha = torch.view_as_complex(xy).angle()
    return torch.stack([beta, alpha], -1)

def pano_to_img_coord(coords):
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-y / np.pi + 0.5, -x / (2. * np.pi) + 0.5], -1)

def direction_to_img_coord(dirs):
    return pano_to_img_coord(direction_to_pano_coord(dirs))

def img_coord_to_sample_coord(coords):
    return torch.stack([coords[..., 1], coords[..., 0]], -1) * 2. - 1.

# Main function
def split_panorama(panorama, gen_res=40, ratio=1.1, device='cpu'):
    """
    Split a panorama into 20 images using an icosphere.
    
    Parameters:
        panorama: torch.Tensor, shape [B, 3, H, W] (RGB panorama image)
        gen_res: int, resolution of generated images for each icosphere face.
    
    Returns:
        pers_imgs: torch.Tensor, shape [20, 3, gen_res, gen_res]
                   20 sub-images corresponding to icosphere faces.
    """
    ico_sphere = IcoSphere(subdivisions=0)
    vertices, faces = torch.tensor(ico_sphere.vertices, dtype=torch.float32), ico_sphere.faces
    ang = np.arctan(.525731112119133606 / .850650808352039932)
    rot_vec = np.array([ang, 0., 0.])
    rot = Rotation.from_rotvec(rot_vec)
    vertices = rot.apply(vertices)
    vertices = vertices.astype(np.float32)

    pers_imgs = []
    # Generate coords for each face
    all_dirs = []
    all_ratios = []
    to_vecs = []
    down_vecs = []
    right_vecs = []

    for face in faces:
        pt_a, pt_b, pt_c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        dirs, ratios, to_vec, down_vec, right_vec = _verts_to_dirs(pt_a, pt_b, pt_c, gen_res=gen_res, ratio=ratio)
        all_dirs.append(dirs)
        all_ratios.append(ratios)
        to_vecs.append(to_vec)
        down_vecs.append(down_vec)
        right_vecs.append(right_vec)

    pers_dirs = torch.stack(all_dirs, 0)
    pers_ratios = torch.stack(all_ratios, 0)
    to_vecs = torch.stack(to_vecs, 0)
    down_vecs = torch.stack(down_vecs, 0)
    right_vecs = torch.stack(right_vecs, 0)
    

    # fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
    # fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
    # cx = torch.ones_like(fx) * gen_res * .5
    # cy = torch.ones_like(fy) * gen_res * .5

    fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * .5
    fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * .5
    cx = torch.ones_like(fx) * .5
    cy = torch.ones_like(fy) * .5
    camera_k = torch.tensor([[[fx[0][0],   0.0000, cx[0][0]],
                              [0.0000, fy[0][0], cy[0][0]],
                              [0.0000,   0.0000,   1.0000]]], 
                            dtype=torch.float32, requires_grad=False, device=device)
    
    rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                            down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                            to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                            dim=1).to(device)
    rot_c2w = rot_w2c.inverse()  # 或使用 rot_w2c.t()
    # quaternion = Rotation.from_matrix(rot_w2c.numpy()).as_quat() 
    # quaternion = torch.tensor(quaternion, dtype=torch.float32, requires_grad=False, device=device)

    # 构建 4x4 的齐次变换矩阵
    extrinsic_matrix_4x4 = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(20,1,1)  # 形状: [4, 4]
    extrinsic_matrix_4x4[:, :3, :3] = rot_c2w
    # print("齐次变换矩阵 (4x4):\n", extrinsic_matrix_4x4)

    n_pers = len(pers_dirs)
    img_coords = direction_to_img_coord(pers_dirs).to(device)
    sample_coords = img_coord_to_sample_coord(img_coords).to(device)
    
    pers_imgs = F.grid_sample(panorama[None].expand(n_pers, -1, -1, -1).to(device), sample_coords, padding_mode='border', align_corners=True)
    # pers_imgs = F.grid_sample(panorama[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]

    return pers_imgs, extrinsic_matrix_4x4, camera_k

# Image I/O functions
def load_panorama(image_path):
    """
    Load a panorama image and convert it to a PyTorch tensor.
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return img_tensor

def save_sub_images(pers_imgs, output_dir):
    """
    Save each sub-image to the specified directory.
    """
    # output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    for i, img_tensor in enumerate(pers_imgs):
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(output_dir + f"ori_perspective_{i + 1}.png")