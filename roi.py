import torch 
import torch.nn as nn 
from typing import Tuple
import torch.nn.functional as F
import math

class ROIPooling(nn.Module): 

    def __init__(self, output_size : Tuple[int, int], scale : float): 
        """
        Implements Region of Interest (ROI) Pooling for object detection from scratch.

        Attributes:
            output_size (Tuple[int, int]): The size of the output feature map after pooling.
            spatial_scale (float): A scaling factor to map the input coordinates to the feature map coordinates.
        """
        super(ROIPooling, self).__init__()

        self.output_size = output_size 
        self.scale = scale
    
    def forward(self, feautre_maps : torch.tensor, ROI : torch.tensor):
        """
        Applies ROI Pooling to the input feature map.

        Args:
            features (torch.Tensor): The input feature map with shape (batch_size, num_channels, height, width).
            rois (torch.Tensor): The ROIs with shape (num_rois, 5), where each ROI is represented as
                                 (batch_index, x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: The pooled feature map with shape (num_rois, num_channels, output_height, output_width).
        """
        height, width = self.output_size 

        num_of_roi = ROI.size(0)
        channels = feautre_maps.size(1)

        output = torch.zeros([num_of_roi, channels, height, width], 
                             dtype = feautre_maps.dtype, device = feautre_maps.device)

        for i in range(num_of_roi): 
            batch_idx, x_min, y_min, x_max, y_max = torch.round(ROI[i] * self.scale).long()
            region_of_interest = feautre_maps[batch_idx, :, y_min:y_max, x_min:x_max]

            roi_height, roi_width = region_of_interest.shape[-2:]

            bin_height = roi_height / height
            bin_width = roi_width / width

            for h_pos in range(height): 
                for w_pos in range(width): 
                    start_x = math.floor(h_pos * bin_width)
                    end_x = math.ceil((h_pos + 1) * bin_width)
                    start_y = math.floor(w_pos * bin_height)
                    end_y = math.ceil((w_pos + 1) * bin_height)

                    bin = region_of_interest[:, start_x:end_x, start_y:end_y]

                    output[i, :, h_pos, w_pos] = torch.max(bin.reshape(512, -1),dim=1)[0]

        return output
    
class ROIAlign(nn.Module): 

    def __init__(self,output_size : Tuple[int, int], scale : float, sampling_ratio : int): 
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.scale = scale 
        self.sampling_ratio = sampling_ratio

    def forward(self, feature_maps : torch.tensor, ROI : torch.tensor): 

        height, width = self.output_size

        num_of_roi = ROI.size(0)
        channels = feature_maps.size(1)
        
        output = torch.zeros(num_of_roi, channels, height, width, 
                             dtype = feature_maps.dtype, device = feature_maps.device)
        
        for i in range(num_of_roi): 
            batch_index, xmin, ymin, xmax, ymax = torch.round(ROI[i] * self.scale).long() 
            region_of_interest = feature_maps[batch_index, :, ymin:ymax, xmin:xmax]

            f_map_height, f_map_width = region_of_interest.shape[-2:]

            bin_height = f_map_height / float(height)
            bin_width = f_map_width / float(width)

            for h_pos in range(height): 
                for w_pos in range(width): 
                    start_x = int(math.floor(w_pos * bin_width))
                    end_x = int(math.ceil((w_pos + 1) * bin_width))
                    start_y = int(math.floor(h_pos * bin_height))
                    end_y = int(math.ceil((h_pos + 1) * bin_height))

                    pool_region = region_of_interest[:, start_y:end_y, start_x:end_x]

                    if pool_region.numel() == 0:
                        output[i, :, h_pos, w_pos] = 0
                    else:
                        output[i, :, h_pos, w_pos] = F.avg_pool2d(
                            pool_region, (end_y - start_y, end_x - start_x), stride=1, padding=0).view(-1)
                        
        return output