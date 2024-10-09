import torch
import torch.nn as nn
from typing import Tuple
from torchvision.ops import roi_align, roi_pool

class ROIPooling(nn.Module):
    """
    Implements Region of Interest (ROI) Pooling as a PyTorch module using torchvision's roi_pool.

    Attributes:
        output_size (Tuple[int, int]): The size (height, width) of the output feature map after pooling.
        spatial_scale (float): A scaling factor to adjust the ROI coordinates.
    """
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float):
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
                (batch_index, x_min, y_min, x_max, y_max).
                Shape: (num_rois, 5)

        Returns:
            torch.Tensor: The pooled feature maps with shape
                (num_rois, num_channels, output_height, output_width)
        """
        # Adjust rois format (batch_index, x_min, y_min, x_max, y_max)
        rois[:, 1:] *= self.spatial_scale

        output = roi_pool(
            feature_maps, rois, self.output_size,
            spatial_scale=self.spatial_scale
        )
        return output

class ROIAlign(nn.Module):
    """
    Implements Region of Interest (ROI) Align as a PyTorch module using torchvision's roi_align.

    Attributes:
        output_size (Tuple[int, int]): The size (height, width) of the output feature map after alignment.
        spatial_scale (float): A scaling factor to adjust the ROI coordinates.
        sampling_ratio (int): The number of sampling points in the interpolation grid.
    """
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float, sampling_ratio: int = 0):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, feature_maps: torch.Tensor, rois: torch.Tensor):
        """
        Applies ROI Align to the selected regions in the feature maps.

        Args:
            feature_maps (torch.Tensor): The input feature maps from the backbone network.
                Shape: (batch_size, num_channels, height, width)
            rois (torch.Tensor): The regions of interest to align, specified as
                (batch_index, x_min, y_min, x_max, y_max).
                Shape: (num_rois, 5)

        Returns:
            torch.Tensor: The aligned feature maps with shape
                (num_rois, num_channels, output_height, output_width)
        """

        # Adjust rois format (batch_index, x_min, y_min, x_max, y_max)
        rois[:, 1:] *= self.spatial_scale

        output = roi_align(
            feature_maps, rois, self.output_size,
            spatial_scale=self.spatial_scale, sampling_ratio=self.sampling_ratio
        )
        return output