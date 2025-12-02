import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

def reshape_normalize(x):
    '''
    Args:
        x: [B, C, H, W]

    Returns:

    '''
    B, C, H, W = x.shape
    x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator==0, 1, denominator)
    return x / denominator

def normalize(x):
    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator == 0, 1, denominator)
    return x / denominator

def single_features_to_RGB(sat_features, idx=0, img_name='test_img.png'):
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    # 1. 重塑特征图形状为 [256, 64*64]
    B, C, H, W = sat_feat.shape
    flatten = np.concatenate([sat_feat], axis=0)
    # 2. 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))
    
    # 3. 归一化到 [0, 1] 范围
    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, H, W, 3)

    sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
    # sat = sat.resize((512, 512))
    sat.save(img_name)

def sat_features_to_RGB(sat_features, grd_features, idx=0):
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    grd_feat = grd_features[idx:idx+1,:,:,:].data.cpu().numpy()
    # 1. 重塑特征图形状为 [256, 64*64]
    B, C, A, A = sat_feat.shape
    _, _, H, W = grd_feat.shape
    flatten = np.concatenate([sat_feat.reshape(B, C, -1), grd_feat.reshape(B, C, -1)], axis=0)
    # 2. 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    pca.fit(normalize(flatten.transpose([0,2,1]).reshape(-1, C)))
    
    # 3. 归一化到 [0, 1] 范围
    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, A, A, 3)
    fuse_feat_new = ((normalize(pca.transform(reshape_normalize(grd_feat))) + 1 )/ 2).reshape(B, H, W, 3)

    sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
    sat = sat.resize((A, A))
    sat.save('sat_feat.png')
    
    grd = Image.fromarray((fuse_feat_new[0] * 255).astype(np.uint8))
    grd = grd.resize((W, H))
    grd.save('grd_feat.png')

def features_to_RGB(sat_feat, g2s_feat_center, g2s_conf_center, g2s_feat_gt, g2s_conf_gt, loop, level, save_dir):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    sat_feat = sat_feat.data.cpu().numpy()  # [B, C, H, W]
    g2s_feat_center = g2s_feat_center.data.cpu().numpy()  # [B, C, H, W]
    g2s_feat_gt = g2s_feat_gt.data.cpu().numpy()

    B, C, A, _ = sat_feat.shape

    flatten = np.concatenate([sat_feat, g2s_feat_center, g2s_feat_gt], axis=0)

    # if level == 0:
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))

    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, A, A, 3)

    mask_center = g2s_conf_center[:, 0, :, :, None].data.cpu().numpy()
    mask_center = mask_center / mask_center.max()
    mask = np.linalg.norm(g2s_feat_center, axis=1)[:, :, :, None] > 0
    g2s_feat_new_center = ((normalize(pca.transform(reshape_normalize(g2s_feat_center))) + 1) / 2).reshape(B, A, A, 3) * mask

    mask_gt = g2s_conf_gt[:, 0, :, :, None].data.cpu().numpy()
    mask_gt = mask_gt / mask_gt.max()
    mask = np.linalg.norm(g2s_feat_gt, axis=1)[:, :, :, None] > 0
    g2s_feat_new_gt = ((normalize(pca.transform(reshape_normalize(g2s_feat_gt))) + 1) / 2).reshape(B, A, A, 3) * mask

    for idx in range(B):
        sat = Image.fromarray((sat_feat_new[idx] * 255).astype(np.uint8))
        sat = sat.resize((512, 512))
        sat.save(os.path.join(save_dir, 'level_' + str(level) + '_sat_feat_' + str(loop * B + idx) + '.png'))

        g2s_center = Image.fromarray((g2s_feat_new_center[idx] * 255).astype(np.uint8))
        g2s_center = g2s_center.resize((512, 512))
        g2s_center.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_center' + str(loop * B + idx) + '.png'))

        g2s_center = Image.fromarray((g2s_feat_new_center[idx] * mask_center[idx] * 255).astype(np.uint8))
        g2s_center = g2s_center.resize((512, 512))
        g2s_center.save(
            os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_center_conf' + str(loop * B + idx) + '.png'))

        g2s_gt = Image.fromarray((g2s_feat_new_gt[idx] * 255).astype(np.uint8))
        g2s_gt = g2s_gt.resize((512, 512))
        g2s_gt.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_gt' + str(loop * B + idx) + '.png'))

        g2s_gt = Image.fromarray((g2s_feat_new_gt[idx] * mask_gt[idx] * 255).astype(np.uint8))
        g2s_gt = g2s_gt.resize((512, 512))
        g2s_gt.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_gt_conf' + str(loop * B + idx) + '.png'))

    return


def pca_2d_hsv_color(pca_2d, H, W):
    """
    将 2D PCA 的结果 (H*W, 2)：
      1. 对 x, y 各自做 min-max 归一化 -> [0,1]
      2. 将 (x, y) 映射到 HSV: H=x, S=y, V=1.0
      3. 转成 RGB，最后 reshape 到 (H, W, 3)
    """
    # pca_2d: shape (H*W, 2)
    pca_2d_norm = pca_2d.copy()

    # 分别对 x, y 做 min-max
    x_min, x_max = pca_2d_norm[:,0].min(), pca_2d_norm[:,0].max()
    y_min, y_max = pca_2d_norm[:,1].min(), pca_2d_norm[:,1].max()
    pca_2d_norm[:,0] = (pca_2d_norm[:,0] - x_min) / (x_max - x_min + 1e-8)
    pca_2d_norm[:,1] = (pca_2d_norm[:,1] - y_min) / (y_max - y_min + 1e-8)

    # HSV: hue = x, saturation = y, value = 1.0
    hsv = np.zeros((pca_2d_norm.shape[0], 3))
    hsv[:, 0] = pca_2d_norm[:,0]       # Hue
    hsv[:, 1] = pca_2d_norm[:,1]       # Saturation
    hsv[:, 2] = 1.0                    # Value=1
    # hsv[:, 2] = 0.7                    # Value=1
    # 转成 RGB
    rgb = mcolors.hsv_to_rgb(hsv)  # (H*W, 3)
    rgb = rgb.reshape(H, W, 3)     # (H, W, 3)
    gamma = 1.2  # >1会让图整体变暗
    rgb = rgb ** (1 / gamma)
    return rgb

def sat_features_to_RGB_2D_PCA(sat_features, grd_features, idx=0):
    """
    1) 取第 idx 个 batch 的 sat_feat, grd_feat
    2) 用 2D PCA 降维 -> (x, y)
    3) 每张图各自 reshape 回原尺寸后，映射到 HSV->RGB
    4) 保存可视化结果
    """
    def reshape_normalize(feat):
        """
        feat: shape (B, C, H, W)
        先把它 reshape 成 (B*H*W, C) 方便 PCA 的 transform。
        """
        B, C, H, W = feat.shape
        # (B, C, H, W) -> (B, C, H*W)
        feat = feat.reshape(B, C, -1)  
        # (B, C, H*W) -> (B, H*W, C)
        feat = feat.transpose(0,2,1)
        # (B, H*W, C) -> (B*H*W, C)
        feat = feat.reshape(-1, C)
        return feat
    # 取第 idx 个的特征
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    grd_feat = grd_features[idx:idx+1,:,:,:].data.cpu().numpy()

    B, C, A, A_ = sat_feat.shape  # A == A_
    _, _, H, W = grd_feat.shape

    # (1) reshape + 合并做 PCA 拟合（确保同一映射）
    sat_flat = reshape_normalize(sat_feat)  # shape: (A*A, C)
    grd_flat = reshape_normalize(grd_feat)  # shape: (H*W, C)
    combined = np.concatenate([sat_flat, grd_flat], axis=0)  # (A*A + H*W, C)

    # (2) 先整体 normalize，再 2D PCA
    combined_norm = normalize(combined)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(combined_norm)

    # 分别 transform
    sat_2d = pca.transform(normalize(sat_flat))  # shape: (A*A, 2)
    grd_2d = pca.transform(normalize(grd_flat))  # shape: (H*W, 2)

    # (3) 映射到 HSV->RGB
    sat_rgb = pca_2d_hsv_color(sat_2d, A, A)
    grd_rgb = pca_2d_hsv_color(grd_2d, H, W)

    # (4) 转成 [0,255] 并保存图像
    sat_img = Image.fromarray((sat_rgb * 255).astype(np.uint8))
    sat_img.save('sat_feat_2dpca.png')

    grd_img = Image.fromarray((grd_rgb * 255).astype(np.uint8))
    grd_img.save('grd_feat_2dpca.png')

    print("Saved sat_feat_2dpca.png and grd_feat_2dpca.png.")


def grd_features_to_RGB_2D_PCA_concat(grd_features, b_idx=0):
    """
    仅针对 grd_features, 形状: (B, V, C, H, W).
    1. 遍历同一个 batch b_idx 下的所有 v_idx -> 得到多张单图
    2. 将它们从上到下拼接成一张图
    3. 保存最终的大图
    """
    B, V, C, H, W = grd_features.shape

    # 判断 b_idx 合法
    assert 0 <= b_idx < B, f"b_idx={b_idx} 超出范围 [0, {B-1}]"

    # 用于存储每个视角的单图 (PIL Image)
    image_list = []

    for v_idx in range(V):
        # 1. 取出第 b_idx 个 batch、第 v_idx 个视角特征
        #    如果 grd_features 在 GPU，需要先 .cpu().numpy()
        feat = grd_features[b_idx, v_idx].detach().cpu().numpy()  # shape (C, H, W)

        # 2. reshape 成 (H*W, C)，然后 normalize
        feat_reshaped = feat.reshape(C, -1).transpose(1, 0)  # (H*W, C)
        feat_norm = normalize(feat_reshaped)

        # 3. 2D PCA
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(feat_norm)  # (H*W, 2)

        # 4. 映射到 HSV->RGB
        rgb = pca_2d_hsv_color(pca_2d, H, W)

        # 5. 转成 [0,255] 并生成 PIL Image
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        image_list.append(img)

    # ---- 所有视角的图像都在 image_list 里了，现在拼接它们 ----

    # 确定拼接后图像的宽度为所有图的最大宽度(一般它们应该相同)
    total_width = max(im.width for im in image_list)
    # 从上到下拼接，高度相加
    total_height = sum(im.height for im in image_list)

    # 建立一个空白画布来放置它们
    concat_img = Image.new("RGB", (total_width, total_height))

    # 逐张贴上去
    y_offset = 0
    for im in image_list:
        concat_img.paste(im, (0, y_offset))
        y_offset += im.height

    # 最终保存
    out_filename = f'grd_feat_2dpca_b{b_idx}_concat.png'
    concat_img.save(out_filename)
    print(f"Saved concatenated image: {out_filename}")


def visualize_1d_pca(tensor1, tensor2, output_filename="pca_visualization.png"):
    """
    使用1D PCA降维并可视化两个[1, 32, 128, 128]的tensor的特征，并将结果保存为.png文件。
    为可视化结果添加颜色映射，并绘制在同一张图上。

    参数:
    tensor1 (torch.Tensor): 第一个输入的tensor，形状为 [1, 32, 128, 128]
    tensor2 (torch.Tensor): 第二个输入的tensor，形状为 [1, 32, 128, 128]
    output_filename (str): 输出图像的文件名，默认为 'pca_visualization.png'
    """
    # 确保输入是四维tensor
    assert tensor1.ndimension() == 4 and tensor1.shape[0] == 1, "输入tensor1必须是[1, 32, 128, 128]的四维tensor"
    assert tensor2.ndimension() == 4 and tensor2.shape[0] == 1, "输入tensor2必须是[1, 32, 128, 128]的四维tensor"

    # 将两个tensor展平为[32, 128*128]
    tensor1_flat = tensor1.view(32, -1)  # 32x(128*128)
    tensor2_flat = tensor2.view(32, -1)  # 32x(128*128)

    # 转换为numpy数组，便于PCA操作
    tensor1_flat_np = tensor1_flat.cpu().detach().numpy()
    tensor2_flat_np = tensor2_flat.cpu().detach().numpy()

    # 使用sklearn中的PCA进行1D降维
    pca = PCA(n_components=1)
    tensor1_pca = pca.fit_transform(tensor1_flat_np)
    tensor2_pca = pca.fit_transform(tensor2_flat_np)

    # 创建一个渐变色的颜色映射
    norm = plt.Normalize(vmin=min(np.min(tensor1_pca), np.min(tensor2_pca)),
                         vmax=max(np.max(tensor1_pca), np.max(tensor2_pca)))
    cmap = cm.viridis  # 你可以选择不同的colormap，如 'viridis', 'plasma', 'inferno', 'magma' 等
    
    # 使用色彩映射给PCA结果着色
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(tensor1_pca)), tensor1_pca, c=tensor1_pca, cmap=cmap, norm=norm, label="Tensor 1", alpha=0.7)
    plt.scatter(np.arange(len(tensor2_pca)), tensor2_pca, c=tensor2_pca, cmap=cmap, norm=norm, label="Tensor 2", alpha=0.7)
    
    # 添加标题和标签
    plt.title("1D PCA Visualization with Colormap")
    plt.xlabel("Channels")
    plt.ylabel("PCA Component Value")

    # 添加颜色条
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), label='PCA Component Value')

    # 添加图例
    plt.legend()

    # 保存图像为.png文件
    plt.savefig(output_filename, format='png')
    print(f"图像已保存为 {output_filename}")
    plt.close()

def single_features_to_RGB_colormap(sat_features, idx=0, img_name='test_img_cmap_zeros_black.png', cmap_name='viridis', zero_threshold=1e-6):
    """
    Visualizes features using the first PCA component and a colormap.
    Pixels where original features are all close to zero are set to black.

    Args:
        sat_features (torch.Tensor or np.ndarray): Feature tensor of shape [B, C, H, W].
        idx (int): Batch index to visualize.
        img_name (str): Output image file name.
        cmap_name (str): Name of the matplotlib colormap to use.
        zero_threshold (float): Threshold below which feature absolute values are considered zero.
    """
    # Helper functions (assuming they exist or define them)
    def reshape_normalize(features):
        """Reshapes [B, C, H, W] to [B*H*W, C] and normalizes features."""
        B, C, H, W = features.shape
        features_reshaped = features.transpose(0, 2, 3, 1).reshape(-1, C)
        # Example normalization (adapt if needed)
        mean = np.mean(features_reshaped, axis=0, keepdims=True)
        std = np.std(features_reshaped, axis=0, keepdims=True)
        std[std == 0] = 1e-6
        normalized = (features_reshaped - mean) / std
        return normalized

    # --- Ensure NumPy array on CPU ---
    if hasattr(sat_features, 'data') and hasattr(sat_features, 'cpu'):
        sat_feat_batch = sat_features.data.cpu().numpy()
    elif isinstance(sat_features, np.ndarray):
        sat_feat_batch = sat_features
    else:
        raise TypeError("Input must be a PyTorch tensor or NumPy array")

    sat_feat = sat_feat_batch[idx:idx+1, :, :, :] # Shape [1, C, H, W]
    B, C, H, W = sat_feat.shape
    assert B == 1

    # --- 0. Identify "Zero" Feature Locations BEFORE Normalization/PCA ---
    # Find pixels where the sum of absolute feature values is below the threshold
    # Reshape to [H, W, C] for easier spatial masking
    sat_feat_spatial = sat_feat[0].transpose(1, 2, 0) # Shape [H, W, C]
    # Check if *all* channels are close to zero for a pixel
    is_zero_mask = np.all(np.abs(sat_feat_spatial) < zero_threshold, axis=-1) # Shape [H, W]
    # Alternatively, check if the norm is close to zero:
    # feature_norm = np.linalg.norm(sat_feat_spatial, axis=-1)
    # is_zero_mask = feature_norm < zero_threshold * np.sqrt(C) # Adjust threshold based on norm


    # --- 1. Prepare data for PCA (Using only non-zero pixels might be better) ---
    # Option A: Use all data (simpler)
    flatten_slice = reshape_normalize(sat_feat)
    # Option B: Use only non-zero data for fitting (potentially more robust PCA)
    # sat_feat_reshaped_orig = sat_feat.transpose(0, 2, 3, 1).reshape(-1, C)
    # non_zero_features = sat_feat_reshaped_orig[~is_zero_mask.reshape(-1)]
    # if non_zero_features.shape[0] < 2: # Need at least 2 samples for PCA
    #     print("Warning: Too few non-zero features for PCA. Saving black image.")
    #     img = Image.fromarray(np.zeros((H,W,3), dtype=np.uint8))
    #     img.save(img_name)
    #     return
    # flatten_slice_nonzero_normalized = reshape_normalize(non_zero_features[np.newaxis,:,:,:]) # Requires adapting reshape_normalize

    # --- 2. PCA (only need 1 component) ---
    pca = PCA(n_components=1)
    # pca.fit(flatten_slice_nonzero_normalized) # Fit on non-zero data if using Option B
    pca.fit(flatten_slice) # Fit on all data (Option A)

    # Transform *all* original slice data (even zeros, though their transform might be less meaningful)
    sat_feat_reshaped = sat_feat.transpose(0, 2, 3, 1).reshape(-1, C)
    pca_transformed_1d = pca.transform(sat_feat_reshaped) # Shape [H*W, 1]

    # --- 3. Normalize the first component to [0, 1] ---
    pc1 = pca_transformed_1d.reshape(H, W) # Reshape to [H, W] first
    # Normalize using only the non-zero pixels' range for better contrast
    pc1_non_zero = pc1[~is_zero_mask]
    if pc1_non_zero.size == 0: # Handle case where all pixels were zero
         normalized_pc1_image = np.zeros((H,W)) + 0.5
    else:
        pc1_min = np.min(pc1_non_zero)
        pc1_max = np.max(pc1_non_zero)
        if pc1_max == pc1_min:
            # If all non-zero pixels map to the same PC1 value, assign a mid-value
             normalized_pc1_image = np.zeros((H,W)) # Start with zeros
             normalized_pc1_image[~is_zero_mask] = 0.5 # Set non-zero pixels to 0.5
        else:
            # Normalize PC1 values based on the range of non-zero pixels
            normalized_pc1 = (pc1 - pc1_min) / (pc1_max - pc1_min)
            # Clamp values potentially outside [0,1] due to extrapolation on zero pixels
            normalized_pc1_image = np.clip(normalized_pc1, 0.0, 1.0)
            # Ensure originally zero pixels don't affect normalization scaling visibly
            normalized_pc1_image[is_zero_mask] = 0.0 # Or assign a value reflecting "background" like 0 or 0.5


    # --- 4. Apply Colormap ---
    try:
        cmap = plt.get_cmap(cmap_name)
        # Apply colormap - cmap expects values in [0, 1]
        colored_image = cmap(normalized_pc1_image)[:, :, :3] # Shape [H, W, 3], range [0, 1]
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis'.")
        cmap = plt.get_cmap('viridis')
        colored_image = cmap(normalized_pc1_image)[:, :, :3]

    # --- 5. Apply Zero Mask ---
    # Where the original features were zero, set the color to black
    # Need to broadcast is_zero_mask [H, W] to [H, W, 3]
    colored_image[is_zero_mask] = 0.0 # Set RGB to (0, 0, 0)

    # --- 6. Convert to uint8 and Save ---
    final_image_uint8 = (colored_image * 255).astype(np.uint8)
    img = Image.fromarray(final_image_uint8)
    # img = img.resize((512, 512)) # Optional resize
    img.save(img_name)
    print(f"Saved colormapped feature visualization (zeros as black) to {img_name}")

def _apply_colormap_and_save(normalized_pc_image, is_zero_mask, cmap_name, img_name):
    """
    应用颜色映射，设置零值掩码区域为黑色，并保存图像。
    """
    try:
        cmap = plt.get_cmap(cmap_name)
        colored_image = cmap(normalized_pc_image)[:, :, :3]
    except ValueError:
        print(f"警告: 颜色映射 '{cmap_name}' 未找到。使用 'viridis'。")
        cmap = plt.get_cmap('viridis')
        colored_image = cmap(normalized_pc_image)[:, :, :3]
    
    colored_image[is_zero_mask] = 0.0
    final_image_uint8 = (colored_image * 255).astype(np.uint8)
    img = Image.fromarray(final_image_uint8)
    img.save(img_name)
    print(f"已将统一颜色映射的可视化结果保存到 {img_name}")


def visualize_two_features_unified_colormap(
    sat_features1, 
    sat_features2, 
    idx=0, 
    img_name_base='feature_viz_unified', 
    cmap_name='PuBuGn', 
    zero_threshold=1e-6,
    pc_low_percentile=2.0,  # 用于PC1归一化的较低百分位数
    pc_high_percentile=98.0 # 用于PC1归一化的较高百分位数
):
    """
    可视化两个形状相同的sat_features特征集，使用统一颜色映射和百分位裁剪增强对比度。

    Args:
        sat_features1 (torch.Tensor or np.ndarray): 第一个特征张量 [B, C, H, W]。
        sat_features2 (torch.Tensor or np.ndarray): 第二个特征张量 [B, C, H, W]。
        idx (int): 批处理索引。
        img_name_base (str): 输出图像文件名的基础。
        cmap_name (str): matplotlib颜色映射名称。
        zero_threshold (float): 特征绝对值低于此值视为零。
        pc_low_percentile (float): 用于PC1归一化的较低百分位数 (0-100)。
        pc_high_percentile (float): 用于PC1归一化的较高百分位数 (0-100)。
                                     设为0和100则等效于标准的最小-最大归一化。
    """
    
    def convert_to_numpy(features_tensor):
        if isinstance(features_tensor, torch.Tensor):
            return features_tensor.detach().cpu().numpy() if hasattr(features_tensor, 'detach') else features_tensor.cpu().numpy()
        elif isinstance(features_tensor, np.ndarray):
            return features_tensor
        else:
            raise TypeError("输入必须是PyTorch张量或NumPy数组")

    def reshape_normalize_slice(features_slice_single):
        _B_slice, _C_slice, _H_slice, _W_slice = features_slice_single.shape
        features_reshaped = features_slice_single.transpose(0, 2, 3, 1).reshape(-1, _C_slice)
        mean = np.mean(features_reshaped, axis=0, keepdims=True)
        std = np.std(features_reshaped, axis=0, keepdims=True)
        std[std == 0] = 1e-6
        normalized = (features_reshaped - mean) / std
        return normalized

    sat_feat_batch1_np = convert_to_numpy(sat_features1)
    sat_feat_batch2_np = convert_to_numpy(sat_features2)

    if sat_feat_batch1_np.shape[1:] != sat_feat_batch2_np.shape[1:]:
        print("警告: 两个特征集的通道、高度或宽度不匹配。")
    B1, C, H, W = sat_feat_batch1_np.shape
    B2, C2, H2, W2 = sat_feat_batch2_np.shape
    if not (C==C2 and H==H2 and W==W2): raise ValueError("特征维度不匹配!")
    if idx >= B1 or idx >= B2: raise ValueError(f"索引 {idx} 超出批次大小")

    sat_feat1_slice = sat_feat_batch1_np[idx:idx+1, :, :, :]
    sat_feat2_slice = sat_feat_batch2_np[idx:idx+1, :, :, :]

    sat_feat1_spatial = sat_feat1_slice[0].transpose(1, 2, 0)
    is_zero_mask1 = np.all(np.abs(sat_feat1_spatial) < zero_threshold, axis=-1)
    sat_feat2_spatial = sat_feat2_slice[0].transpose(1, 2, 0)
    is_zero_mask2 = np.all(np.abs(sat_feat2_spatial) < zero_threshold, axis=-1)

    flat_norm_feat1 = reshape_normalize_slice(sat_feat1_slice)
    flat_norm_feat2 = reshape_normalize_slice(sat_feat2_slice)
    combined_flat_norm_feat = np.concatenate((flat_norm_feat1, flat_norm_feat2), axis=0)

    if combined_flat_norm_feat.shape[0] < 2:
        print(f"警告: 组合特征样本数 ({combined_flat_norm_feat.shape[0]}) 过少无法PCA。")
        dummy_norm_img = np.zeros((H, W)) + 0.5
        _apply_colormap_and_save(dummy_norm_img, np.ones((H,W),dtype=bool), cmap_name, f"{img_name_base}_feat1.png")
        _apply_colormap_and_save(dummy_norm_img, np.ones((H,W),dtype=bool), cmap_name, f"{img_name_base}_feat2.png")
        return
    
    n_pca_components = min(1, combined_flat_norm_feat.shape[0], combined_flat_norm_feat.shape[1])
    if n_pca_components < 1:
        print(f"警告: 无法确定PCA成分数。")
        # (处理同上)
        dummy_norm_img = np.zeros((H, W)) + 0.5
        _apply_colormap_and_save(dummy_norm_img, np.ones((H,W),dtype=bool), cmap_name, f"{img_name_base}_feat1.png")
        _apply_colormap_and_save(dummy_norm_img, np.ones((H,W),dtype=bool), cmap_name, f"{img_name_base}_feat2.png")
        return
        
    pca = PCA(n_components=n_pca_components)
    pca.fit(combined_flat_norm_feat)

    pc1_transformed_flat_feat1 = pca.transform(flat_norm_feat1)
    pc1_transformed_flat_feat2 = pca.transform(flat_norm_feat2)
    pc1_feat1 = pc1_transformed_flat_feat1.reshape(H, W)
    pc1_feat2 = pc1_transformed_flat_feat2.reshape(H, W)

    # --- 4. 对PC1值进行全局归一化 (使用百分位裁剪) ---
    pc1_non_zero_values_feat1 = pc1_feat1[~is_zero_mask1]
    pc1_non_zero_values_feat2 = pc1_feat2[~is_zero_mask2]
    
    all_pc1_non_zero_values = np.array([])
    if pc1_non_zero_values_feat1.size > 0:
        all_pc1_non_zero_values = np.concatenate((all_pc1_non_zero_values, pc1_non_zero_values_feat1.flatten()))
    if pc1_non_zero_values_feat2.size > 0:
        all_pc1_non_zero_values = np.concatenate((all_pc1_non_zero_values, pc1_non_zero_values_feat2.flatten()))

    normalized_pc1_image1 = np.zeros((H, W))
    normalized_pc1_image2 = np.zeros((H, W))

    if all_pc1_non_zero_values.size == 0:
        normalized_pc1_image1[:] = 0.5
        normalized_pc1_image2[:] = 0.5
        print("警告: 两个特征图在非零区域的PC1值均为空。图像将为单色。")
    else:
        val_min = np.percentile(all_pc1_non_zero_values, pc_low_percentile)
        val_max = np.percentile(all_pc1_non_zero_values, pc_high_percentile)

        if val_max <= val_min: #百分位裁剪后范围为0或负(例如数据非常平坦或百分位选择极端)
            # 回退到使用实际的最小/最大值
            val_min = np.min(all_pc1_non_zero_values)
            val_max = np.max(all_pc1_non_zero_values)
            if val_max <= val_min: # 如果实际最小/最大值也无法提供范围 (数据完全平坦)
                print(f"警告: 所有非零区域的PC1值均相同 (值为 {val_min:.3f})。非零区域将映射到0.5。")
                # 对于这种情况，非零区域统一映射到0.5
                normalized_pc1_image1[~is_zero_mask1] = 0.5
                normalized_pc1_image2[~is_zero_mask2] = 0.5
                # 后续的除法操作不会执行
            else: # 实际最小/最大值可用
                print(f"信息: 百分位裁剪 [{pc_low_percentile}%, {pc_high_percentile}%] 无效或导致范围为零。"
                      f"回退使用实际全局PC1范围 [{val_min:.3f}, {val_max:.3f}] 进行归一化。")
        else: # 百分位裁剪提供了有效范围
             print(f"信息: 使用全局PC1百分位裁剪范围 [{val_min:.3f}, {val_max:.3f}] (基于 {pc_low_percentile}% 和 {pc_high_percentile}% 百分位) 进行归一化。")


        # 使用确定的 val_min 和 val_max 进行归一化
        # 只有在 val_max > val_min 时才进行除法归一化
        if val_max > val_min:
            norm_vals1 = (pc1_feat1 - val_min) / (val_max - val_min)
            norm_vals2 = (pc1_feat2 - val_min) / (val_max - val_min)
            
            normalized_pc1_image1 = np.clip(norm_vals1, 0.0, 1.0)
            normalized_pc1_image2 = np.clip(norm_vals2, 0.0, 1.0)
        # else 分支 (val_max <= val_min) 已在上面处理了平坦数据的情况，
        # normalized_pc1_image1/2 在这种情况下会保持为0，然后非零区域被设为0.5

    img_name1 = f"{img_name_base}_feat1.png"
    img_name2 = f"{img_name_base}_feat2.png"

    _apply_colormap_and_save(normalized_pc1_image1, is_zero_mask1, cmap_name, img_name1)
    _apply_colormap_and_save(normalized_pc1_image2, is_zero_mask2, cmap_name, img_name2)