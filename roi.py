import torch 
import torch.nn as nn 
from typing import Tuple
import torch.nn.functional as F
import math

class ROIPooling(nn.Module):
    """
    Implements Region of Interest (ROI) Pooling as a PyTorch module.

    Attributes:
        output_size (Tuple[int, int]): The size (height, width) of the output feature map after pooling.
        spatial_scale (float): A scaling factor to adjust the ROI coordinates.
    """
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float):
        """
        Initializes the ROIPooling module.

        Args:
            output_size (Tuple[int, int]): The target size (height, width) of the output feature maps.
            spatial_scale (float): The factor to scale the ROI coordinates.
        """
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_maps: torch.Tensor, rois: torch.Tensor):
        """
        Applies ROI Pooling to the selected regions in the feature maps.

        Args:
            feature_maps (torch.Tensor): The input feature maps from the backbone network.
                Shape: (batch_size, num_channels, height, width)
            rois (torch.Tensor): The regions of interest to pool over, specified as
                (x_min, y_min, x_max, y_max, batch_index).
                Shape: (num_rois, 5)

        Returns:
            torch.Tensor: The pooled feature maps with shape
                (num_rois, num_channels, output_height, output_width)
        """
        num_rois = rois.size(0)
        channels = feature_maps.size(1)
        pooled_height, pooled_width = self.output_size
        output = torch.zeros(num_rois, channels, pooled_height, pooled_width, dtype=feature_maps.dtype, device=feature_maps.device)

        for i in range(num_rois):
            x_min, y_min, x_max, y_max, batch_idx = rois[i] * self.spatial_scale
            batch_idx = int(batch_idx)

            x_min, x_max = torch.round(torch.tensor([x_min, x_max]) * self.spatial_scale).long().clamp(0, feature_maps.size(3) - 1)
            y_min, y_max = torch.round(torch.tensor([y_min, y_max]) * self.spatial_scale).long().clamp(0, feature_maps.size(2) - 1)

            x_max = max(x_max, x_min + 1)
            y_max = max(y_max, y_min + 1)

            roi_feature_map = feature_maps[batch_idx, :, y_min:y_max, x_min:x_max]
            if roi_feature_map.numel() > 0:
                output[i] = F.adaptive_max_pool2d(roi_feature_map, self.output_size)
            else:
                output[i] = torch.zeros(channels, pooled_height, pooled_width, dtype=feature_maps.dtype, device=feature_maps.device)

        return output
 
class ROIAlign(nn.Module):
    """
    Implements Region of Interest (ROI) Align as a PyTorch module.

    Attributes:
        output_size (Tuple[int, int]): The size (height, width) of the output feature map after alignment.
        spatial_scale (float): A scaling factor to adjust the ROI coordinates.
    """
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float):
        """
        Initializes the ROIAlign module.

        Args:
            output_size (Tuple[int, int]): The target size (height, width) of the output feature maps.
            spatial_scale (float): The factor to scale the ROI coordinates.
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_maps: torch.Tensor, rois: torch.Tensor):
        """
        Applies ROI Align to the selected regions in the feature maps.

        Args:
            feature_maps (torch.Tensor): The input feature maps from the backbone network.
                Shape: (batch_size, num_channels, height, width)
            rois (torch.Tensor): The regions of interest to align, specified as
                (x_min, y_min, x_max, y_max, batch_index).
                Shape: (num_rois, 5)

        Returns:
            torch.Tensor: The aligned feature maps with shape
                (num_rois, num_channels, output_height, output_width)
        """
        num_rois = rois.size(0)
        channels = feature_maps.size(1)
        pooled_height, pooled_width = self.output_size
        output = torch.zeros(num_rois, channels, pooled_height, pooled_width, dtype=feature_maps.dtype, device=feature_maps.device)

        for i in range(num_rois):
            x_min, y_min, x_max, y_max, batch_idx = rois[i] * self.spatial_scale
            batch_idx = int(batch_idx)

            # Convert coordinates to the spatial scale of the feature map
            x_min, y_min, x_max, y_max = torch.tensor([x_min, y_min, x_max, y_max]) * self.spatial_scale
            x_min = x_min.clamp(0, feature_maps.size(3) - 1)
            y_min = y_min.clamp(0, feature_maps.size(2) - 1)
            x_max = x_max.clamp(x_min + 1, feature_maps.size(3))
            y_max = y_max.clamp(y_min + 1, feature_maps.size(2))

            # Calculate the affine transformation matrix for the ROI
            theta = torch.tensor([[[x_max - x_min, 0, 2 * x_min / feature_maps.size(3) - 1],
                                   [0, y_max - y_min, 2 * y_min / feature_maps.size(2) - 1]]],
                                 device=feature_maps.device).float()

            # Create a grid for the ROI and apply grid sampling
            grid = F.affine_grid(theta, [1, channels, pooled_height, pooled_width], align_corners=False)
            roi_feature_map = feature_maps[batch_idx, :, :].unsqueeze(0)
            output[i] = F.grid_sample(roi_feature_map, grid, align_corners=False).squeeze(0)

        return output
