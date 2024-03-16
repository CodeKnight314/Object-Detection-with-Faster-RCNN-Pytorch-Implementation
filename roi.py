import torch 
import torch.nn as nn 
from typing import Tuple
import torch.nn.functional as F
import math

class ROIPooling(nn.Module):
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float):
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_maps: torch.Tensor, rois: torch.Tensor):
        num_rois = rois.size(0)
        channels = feature_maps.size(1)
        pooled_height, pooled_width = self.output_size
        output = torch.zeros(num_rois, channels, pooled_height, pooled_width, dtype=feature_maps.dtype, device=feature_maps.device)

        for i in range(num_rois):
            x_min, y_min, x_max, y_max, batch_idx = rois[i] * self.spatial_scale
            batch_idx = int(batch_idx)

            x_min, x_max = torch.round(torch.tensor([x_min, x_max]) * self.spatial_scale).long().clamp(0, feature_maps.size(3) - 1)
            y_min, y_max = torch.round(torch.tensor([y_min, y_max]) * self.spatial_scale).long().clamp(0, feature_maps.size(2) - 1)

            x_max = max(x_max, x_min + 1)  # Ensure non-zero region size
            y_max = max(y_max, y_min + 1)  # Ensure non-zero region size

            roi_feature_map = feature_maps[batch_idx, :, y_min:y_max, x_min:x_max]
            if roi_feature_map.numel() > 0:
                output[i] = F.adaptive_max_pool2d(roi_feature_map, self.output_size)
            else:
                output[i] = torch.zeros(channels, pooled_height, pooled_width, dtype=feature_maps.dtype, device=feature_maps.device)

        return output
    
class ROIAlign(nn.Module):
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_maps: torch.Tensor, rois: torch.Tensor):
        num_rois = rois.size(0)
        channels = feature_maps.size(1)
        pooled_height, pooled_width = self.output_size
        output = torch.zeros(num_rois, channels, pooled_height, pooled_width, dtype=feature_maps.dtype, device=feature_maps.device)

        for i in range(num_rois):
            x_min, y_min, x_max, y_max, batch_idx = rois[i] * self.spatial_scale
            batch_idx = int(batch_idx)

            theta = torch.tensor([[[x_max - x_min, 0, x_min * 2.0 / feature_maps.size(3) - 1.0],
                                   [0, y_max - y_min, y_min * 2.0 / feature_maps.size(2) - 1.0]]], device=feature_maps.device)
            
            grid_size = [1, channels, pooled_height, pooled_width]

            grid = F.affine_grid(theta, grid_size, align_corners=False)
            roi_feature_map = feature_maps[batch_idx, :, :].unsqueeze(0)

            output[i] = F.grid_sample(roi_feature_map, grid, align_corners=False).squeeze(0)

        return output