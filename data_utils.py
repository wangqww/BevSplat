
import numpy as np
import torch
import einops as E
import torch.nn.functional as F
from typing import Optional

CameraGPS_shift = [1.08, 0.26]
Satmap_zoom = 18
Camera_height = 1.65 #meter
Camera_distance = 0.54 #meter

SatMap_original_sidelength = 512 # 0.2 m per pixel
SatMap_process_sidelength = 512 # 0.2 m per pixel
Default_lat = 49.015

CameraGPS_shift_left = [1.08, 0.26]
CameraGPS_shift_right = [1.08, 0.8]  # 0.26 + 0.54

EPS = 1e-7

def get_satmap_zoom():
    return Satmap_zoom

def get_camera_height():
    return Camera_height

def get_camera_distance():
    return Camera_distance

def get_original_satmap_sidelength():
    return SatMap_original_sidelength

def get_process_satmap_sidelength():
    return SatMap_process_sidelength

# x: east shift in meter, y: south shift in meter
# return lat and lon after shift
# Curvature formulas from https://en.wikipedia.org/wiki/Earth_radius#Meridional
def meter2latlon(lat, lon, x, y):
    r = 6378137 # equatorial radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    coslat = np.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)
    lon += x / kx 
    lat -= y / ky
    
    return lat, lon   

def gps2meters(lat_s, lon_s, lat_d, lon_d ):
    r = 6378137 # equatorial radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    lat = (lat_s+lat_d)/2
    coslat = np.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)
    x = (lon_d-lon_s)*kx
    y = (lat_s-lat_d)*ky # y: from top to bottom
    
    return [x,y]


def gps2utm(lat, lon, lat0=49.015):
    # from paper "Vision meets Robotics: The KITTI Dataset"

    r = 6378137.
    s = np.cos(lat0 * np.pi / 180)

    x = s * r * np.pi * lon / 180
    y = s * r * np.log(np.tan(np.pi * (90 + lat) / 360))

    return x, y

def gps2utm_torch(lat, lon, lat0=torch.tensor(49.015)):
    # from paper "Vision meets Robotics: The KITTI Dataset"

    r = 6378137.
    s = torch.cos(lat0 * np.pi / 180)

    x = s * r * np.pi * lon / 180
    y = s * r * torch.log(torch.tan(np.pi * (90 + lat) / 360))

    return x, y


def gps2meters_torch(lat_s, lon_s, lat_d, lon_d):
    # inputs: torch array: [n]
    r = 6378137 # equatorial radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    lat = lat_d
    coslat = np.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)
    
    x = (lon_d-lon_s)*kx
    y = (lat_s-lat_d)*ky # y: from top to bottom
    
    return x,y


def gps2shiftmeters(latlon ):
    # torch array: [B,S,2]

    r = 6378137 # equatoristereoal radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    lat = latlon[0,0,0]
    coslat = torch.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = torch.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)

    shift_x = (latlon[:,:1,1]-latlon[:,:,1])*kx #B,S east
    shift_y = (latlon[:,:,0]-latlon[:,:1,0])*ky #B,S south
    shift = torch.cat([shift_x.unsqueeze(-1),shift_y.unsqueeze(-1)],dim=-1) #[B,S,2] #shift from 0
    
    # shift from privious
    S = latlon.size()[1]
    shift = shift[:,1:,:]-shift[:,:(S-1),:]
    
    return shift


def gps2distance(lat_s, lon_s, lat_d, lon_d ):
    x,y = gps2meters_torch(lat_s, lon_s, lat_d, lon_d )
    dis = torch.sqrt(torch.pow(x, 2)+torch.pow(y,2))
    return dis


def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel


def gps2shiftscale(latlon):
    # torch array: [B,S,2]
    
    shift = gps2shiftmeters(latlon)
    
    # turn meter to -1~1
    meter_per_pixel = get_meter_per_pixel(scale=1)
    win_range = meter_per_pixel*SatMap_original_sidelength
    shift /= win_range//2
    
    return shift

def get_camera_max_meter_shift():
    return np.linalg.norm(CameraGPS_shift)

def get_camera_gps_shift(heading):
    shift_x = CameraGPS_shift[0] * np.cos(heading%(2*np.pi)) + CameraGPS_shift[1] * np.sin(heading%(2*np.pi))
    shift_y = CameraGPS_shift[1] * np.cos(heading%(2*np.pi)) - CameraGPS_shift[0] * np.sin(heading%(2*np.pi))
    return shift_x, shift_y


def get_camera_gps_shift_left(heading):
    shift_x = CameraGPS_shift_left[0] * np.cos(heading%(2*np.pi)) + CameraGPS_shift_left[1] * np.sin(heading%(2*np.pi))
    shift_y = CameraGPS_shift_left[0] * np.sin(heading%(2*np.pi)) - CameraGPS_shift_left[1] * np.cos(heading%(2*np.pi))
    return shift_x, shift_y


def get_camera_gps_shift_right(heading):
    shift_x = CameraGPS_shift_right[0] * np.cos(heading%(2*np.pi)) + CameraGPS_shift_right[1] * np.sin(heading%(2*np.pi))
    shift_y = CameraGPS_shift_right[0] * np.sin(heading%(2*np.pi)) - CameraGPS_shift_right[1] * np.cos(heading%(2*np.pi))
    return shift_x, shift_y


def get_height_config():
    start = 0 #-15 -7 0
    end = 0 
    count = 1 #16 8 1
    return start, end, count

    
def center_padding(images, patch_size):
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

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

def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x, y = torch.meshgrid(
        [
            torch.arange(orig_x, w + orig_x, step_x, device=device),
            torch.arange(orig_y, h + orig_y, step_y, device=device),
        ],
        indexing="xy",
    )
    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)
    R = torch.tensor([[0, 1], [1, 0]]).float().to(grid.device)
    XZ = torch.einsum('ij, hwj -> hwi', R, grid)  # shape = [satmap_sidelength, satmap_sidelength, 2]
    return XZ

def from_homogeneous(points, eps: float = 1e-8):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)