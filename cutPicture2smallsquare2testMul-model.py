import cv2
import numpy as np
import os
import pandas as pd

# 参数设定
image_path = 'data/LN/FF256/001FF/001ff_0_0_0_0.png'
gt_image_path = 'data/LN/FF256/001FF/001ff_0_0_0_0_gt.png'
pixel_features_file = 'H_SLIC.npy'
output_dir = 'data/LN/FF256/001FF/patches/'

image_size = 256
num_patches = 4
patch_size = image_size // num_patches

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 加载图像和特征
image = cv2.imread(image_path)
gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
pixel_features = np.load(pixel_features_file)

# 初始化特征列表
segment_features = []

# 遍历每个小图片并保存
for i in range(num_patches):
    for j in range(num_patches):
        x_start = i * patch_size
        x_end = x_start + patch_size
        y_start = j * patch_size
        y_end = y_start + patch_size

        # 裁剪小图片和真值图
        patch_image = image[x_start:x_end, y_start:y_end]
        patch_gt_image = gt_image[x_start:x_end, y_start:y_end]

        # 裁剪特征
        patch_pixel_features = pixel_features[x_start:x_end, y_start:y_end]

        # 保存小图片和特征
        patch_image_filename = f'patch_{i}_{j}.png'
        patch_gt_image_filename = f'patch_{i}_{j}_gt.png'
        patch_features_filename = f'patch_{i}_{j}_features.npy'

        cv2.imwrite(os.path.join(output_dir, patch_image_filename), patch_image)
        cv2.imwrite(os.path.join(output_dir, patch_gt_image_filename), patch_gt_image)
        np.save(os.path.join(output_dir, patch_features_filename), patch_pixel_features)

        # 计算真值为1（养殖区域）的像素数量
        num_fish_pixels = np.sum(patch_gt_image == 255)

        # 计算当前超像素区域中真值为1的比例
        fish_ratio = num_fish_pixels / np.sum(patch_gt_image >= 0)

        # 准备保存到CSV文件的数据
        segment_data = {
            'imageid': f'image_{i}_{j}',
            'geographic_coordinate_system': 'GCS WGS 1984',
            'image_id_segment_label': f'{i}_{j}',
            'image_class': '浮筏',
            'fish_ratio': fish_ratio,
            'color_hist': np.random.rand(256).tolist(),
            'mean_pixel_feature': os.path.join(output_dir, patch_features_filename),  # 使用文件路径
            'up': '39.511930',
            'left': '122.891058',
            'right': '122.916569',
            'down': '39.489298',
            'x_min': x_start,
            'y_min': y_start,
            'x_max': x_end,
            'y_max': y_end
        }

        segment_features.append(segment_data)

# 将特征列表保存为CSV文件
features_df = pd.DataFrame(segment_features)
csv_file = 'data/LN/FF256/001FF/CSV/segment_4×4.csv'
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
features_df.to_csv(csv_file, index=False)

print(f"超像素特征已保存到 {csv_file} 文件中。")
