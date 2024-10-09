import torch
import torch.nn as nn
from typing import Tuple
from AnchorGenerator import AnchorGenerator
from utils.box_utils import *
from Proposal_Filter import ProposalFilter

def bbox_adjust(anchors: torch.Tensor, anchor_offsets: torch.Tensor) -> torch.Tensor:
    """
    Adjusts the bounding box deltas and applies them to the anchor boxes.

    Args:
        anchors (torch.Tensor): The anchor boxes tensor with shape (batch_size, num_anchors, 4).
            Each anchor box is represented as (x_min, y_min, x_max, y_max).
        anchor_offsets (torch.Tensor): The predicted bounding box deltas with shape (batch_size, num_anchors, 4).
            Each delta is represented as (dx, dy, dw, dh).

    Returns:
        torch.Tensor: The adjusted anchor boxes tensor with shape (batch_size, num_anchors, 4).
            Each adjusted box is represented as (x_min, y_min, x_max, y_max).
    """
    anchor_widths = anchors[:, :, 2] - anchors[:, :, 0]
    anchor_heights = anchors[:, :, 3] - anchors[:, :, 1]

    anchor_center_x = anchors[:, :, 0] + 0.5 * anchor_widths
    anchor_center_y = anchors[:, :, 1] + 0.5 * anchor_heights

    new_center_x = anchor_offsets[:, :, 0] * anchor_widths + anchor_center_x
    new_center_y = anchor_offsets[:, :, 1] * anchor_heights + anchor_center_y

    new_widths = torch.exp(anchor_offsets[:, :, 2]) * anchor_widths
    new_heights = torch.exp(anchor_offsets[:, :, 3]) * anchor_heights

    top_left_x = new_center_x - 0.5 * new_widths
    top_left_y = new_center_y - 0.5 * new_heights
    bottom_right_x = new_center_x + 0.5 * new_widths
    bottom_right_y = new_center_y + 0.5 * new_heights

    adjusted_anchors = torch.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim=2)

    return adjusted_anchors

def bbox_deltas(anchor_batch_A: torch.Tensor, anchor_batch_B: torch.Tensor) -> torch.Tensor:
    """
    Encodes the relative deltas between two batches of anchor boxes.

    Args:
        anchor_batch_A (torch.Tensor): The first batch of anchor boxes with shape (batch_size, num_anchors, 4).
            Each anchor box is represented as (x_min, y_min, x_max, y_max).
        anchor_batch_B (torch.Tensor): The second batch of anchor boxes with shape (batch_size, num_anchors, 4).
            Each anchor box is represented as (x_min, y_min, x_max, y_max).

    Returns:
        torch.Tensor: The relative deltas tensor with shape (batch_size, num_anchors, 4).
            Each delta is represented as (dx, dy, dw, dh).
    """
    widths_A = anchor_batch_A[:, :, 2] - anchor_batch_A[:, :, 0]
    heights_A = anchor_batch_A[:, :, 3] - anchor_batch_A[:, :, 1]
    widths_B = anchor_batch_B[:, :, 2] - anchor_batch_B[:, :, 0]
    heights_B = anchor_batch_B[:, :, 3] - anchor_batch_B[:, :, 1]

    center_x_A = anchor_batch_A[:, :, 0] + 0.5 * widths_A
    center_y_A = anchor_batch_A[:, :, 1] + 0.5 * heights_A
    center_x_B = anchor_batch_B[:, :, 0] + 0.5 * widths_B
    center_y_B = anchor_batch_B[:, :, 1] + 0.5 * heights_B

    dx = (center_x_B - center_x_A) / widths_A
    dy = (center_y_B - center_y_A) / heights_A
    dw = torch.log(widths_B / widths_A)
    dh = torch.log(heights_B / heights_A)

    deltas = torch.stack([dx, dy, dw, dh], dim=2)

    return deltas

class RPN_head(nn.Module):
    """
    Determines Objectness Score and BBox Regression

    Attributes:
        input_dimensions (int): Number of input channels.
        mid_channels (int): Number of channels in the intermediate convolutional layers.
        num_anchors (int): Number of anchors per location.
        conv_depth (int): Depth of the convolutional layers in the RPN head.
    """
    def __init__(self, input_dimensions: int, mid_channels: int, num_anchors: int, conv_depth: int) -> None:
        super(RPN_head, self).__init__()
        self.input_dimensions = input_dimensions
        self.num_anchors = num_anchors
        self.conv_depth = conv_depth

        self.conv = nn.Sequential(*[nn.Conv2d(input_dimensions, mid_channels, kernel_size=3, stride=1, padding=1) if _ == 0 else
                                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1) for _ in range(conv_depth)])

        self.cls = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1, stride=1, padding=0)
        self.bbox = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RPN head.

        Args:
            x (torch.Tensor): Input feature map with shape (batch_size, input_dimensions, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing objectness scores with shape (batch_size, num_anchors * height * width, 2)
                                              and predicted bounding box deltas with shape (batch_size, num_anchors * height * width, 4).
        """
        N, C, H, W = x.shape
        out = self.conv(x)
        cls_scores = torch.sigmoid(self.cls(out)).reshape(N, -1, 2)
        predicted_bbox = self.bbox(out).permute(0, 2, 3, 1).reshape(N, -1, 4)

        return (cls_scores, predicted_bbox)

class Regional_Proposal_Network(nn.Module):
    """
    Regional Proposal Network (RPN) for generating object proposals.

    Attributes:
        input_dimension (int): Number of input channels.
        mid_dimension (int): Number of channels in the intermediate convolutional layers.
        conv_depth (int): Depth of the convolutional layers in the RPN head.
        score_threshold (float): Score threshold for filtering proposals.
        nms_threshold (float): IoU threshold for non-maximum suppression.
        min_size (int): Minimum size of proposals to be considered.
        max_proposals (int): Maximum number of proposals to be kept after NMS.
        size (Tuple[int]): Tuple of anchor sizes.
        aspect_ratio (Tuple[int]): Tuple of anchor aspect ratios.
    """
    def __init__(self,
                 input_dimension: int,
                 mid_dimension: int,
                 conv_depth: int,
                 score_threshold: float,
                 iou_threshold: float,
                 min_size: int,
                 max_proposals: int,
                 size: Tuple[int],
                 aspect_ratio: Tuple[int],
                 train_mode: bool = True
                 ):

        super(Regional_Proposal_Network, self).__init__()

        self.sizes = size
        self.aspect_ratios = aspect_ratio
        self.num_anchors = len(size) * len(aspect_ratio)
        self.anchor_gen = AnchorGenerator(sizes=size, aspect_ratios=aspect_ratio)

        self.rpn_head = RPN_head(input_dimensions=input_dimension, mid_channels=mid_dimension, num_anchors=self.num_anchors, conv_depth=conv_depth)

        self.proposal_Filter = ProposalFilter(iou_threshold=iou_threshold, min_size=min_size, score_threshold=score_threshold, max_proposals=max_proposals)

        self.train_mode = train_mode

    def forward(self, image_list: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Regional Proposal Network.

        Args:
            image_list (torch.Tensor): List of input images with shape (batch_size, channels, height, width).
            feature_map (torch.Tensor): Feature map from the backbone network with shape (batch_size, input_dimension, feature_map_height, feature_map_width).

        Returns:
            torch.Tensor: Filtered proposals after applying NMS with shape (num_filtered_proposals, 5).
        """
        predict_cls, predict_bbox_deltas = self.rpn_head(feature_map)
        anchors = self.anchor_gen(image_list, feature_map)
        
        decoded_anchors = bbox_adjust(anchors, predict_bbox_deltas)
        
        filtered_anchors, filtered_cls = self.proposal_Filter(decoded_anchors, predict_cls)
        
        roi = filtered_anchors.view(-1, 4)

        batch_index = torch.cat([torch.full((filtered_anchors[i].size(0), 1), i) for i in range(len(filtered_anchors))], dim=0).to(feature_map.device)

        roi = torch.cat([roi, batch_index], dim=1)

        if self.train_mode: 
            return roi, predict_cls, predict_bbox_deltas, anchors
        else:
            return roi