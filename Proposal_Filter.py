import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Tuple
from AnchorGenerator import AnchorGenerator
from utils.box_utils import *
from roi import ROIPooling
from Proposal_Filter import ProposalFilter

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
    super(RPN_head,self).__init__()
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
                 nms_threshold: float,
                 min_size: int,
                 max_proposals: int,
                 size: Tuple[int],
                 aspect_ratio: Tuple[int],
                 ):

        super(Regional_Proposal_Network, self).__init__()

        self.sizes = size
        self.aspect_ratios = aspect_ratio
        self.num_anchors = len(size) * len(aspect_ratio)
        self.anchor_gen = AnchorGenerator(sizes=size, aspect_ratios=aspect_ratio)

        self.rpn_head = RPN_head(input_dimensions=input_dimension, mid_channels=mid_dimension, num_anchors=self.num_anchors, conv_depth=conv_depth)

        self.proposal_Filter = ProposalFilter(iou_threshold=score_threshold, min_size=min_size)

    def forward(self, image_list: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Regional Proposal Network.

        Args:
            image_list (torch.Tensor): List of input images with shape (batch_size, channels, height, width).
            feature_map (torch.Tensor): Feature map from the backbone network with shape (batch_size, input_dimension, feature_map_height, feature_map_width).

        Returns:
            torch.Tensor: Filtered proposals after applying NMS with shape (batch_size, num_filtered_proposals, 4).
        """
        predict_cls, predict_bbox_deltas = self.rpn_head(feature_map)
        anchors = self.anchor_gen(image_list, feature_map)
        
        decoded_anchors = bbox_encode(predict_bbox_deltas, anchors)
        
        filtered_anchors = self.proposal_Filter(decoded_anchors, predict_cls)

        roi = torch.cat(filtered_anchors, dim = 0)
        batch_index = torch.cat([torch.full((len(batch), 1), i) for i, batch in enumerate(filtered_anchors)], dim = 0).to(feature_map.device)
        roi = torch.cat([roi, batch_index], dim = 1)
                
        return roi