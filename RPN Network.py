import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Tuple
from AnchorGenerator import AnchorGenerator
from utils.box_utils import *
from roi import ROIPooling

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

class ProposalFilter(nn.Module):
    """
    Filters proposals based on objectness scores and applies Non-Maximum Suppression (NMS).

    Attributes:
        iou_threshold (float): IoU threshold for NMS.
        min_size (int): Minimum size of proposals to be considered.
    """
    def __init__(self, iou_threshold: float, min_size: int) -> None:
        super(ProposalFilter, self).__init__()
        self.iou_threshold = iou_threshold
        self.min_size = min_size

    def forward(self, proposals: torch.Tensor, cls_scores: torch.Tensor) -> torch.Tensor:
        """
        Filters proposals based on objectness scores and applies NMS.

        Args:
            proposals (torch.Tensor): Proposed bounding boxes with shape (batch_size, num_proposals, 4).
            cls_scores (torch.Tensor): Classification scores for each proposal with shape (batch_size, num_proposals, 2).

        Returns:
            torch.Tensor: Filtered proposals after applying NMS with shape (batch_size, num_filtered_proposals, 4).
        """
        # Filter out proposals with small sizes
        filtered_proposals = []

        for i in range(proposals.shape[0]):
          # Grabs proposals and objectness score from each batch
          prop = proposals[i]
          scores = cls_scores[i]

          # Apply NMS
          filtered_proposals.append(self.nms(prop, scores))

        return filtered_proposals

    def nms(self, proposals: torch.Tensor, cls_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping proposals.

        Args:
            proposals (torch.Tensor): Proposed bounding boxes with shape (num_proposals, 4).
            cls_scores (torch.Tensor): Classification scores for each proposal with shape (num_proposals, 2).

        Returns:
            torch.Tensor: Filtered proposals after applying NMS with shape (num_filtered_proposals, 4).
        """
        objectness_scores = cls_scores[:, 1]
        sorted_scores, sorted_indexes = torch.sort(objectness_scores, descending=True)
        sorted_proposals = proposals[sorted_indexes]

        keep = []
        while sorted_proposals.shape[0] > 0:
            current_proposal = sorted_proposals[0:1]  # Keep dimensions for broadcasting
            keep.append(sorted_indexes[0].item())

            # Calculate IoU between the current proposal and the rest
            ious = self.calculate_iou(current_proposal, sorted_proposals[1:])

            # Keep only proposals with IoU lower than the threshold
            keep_mask = ious < self.iou_threshold
            sorted_proposals = sorted_proposals[1:][keep_mask]
            sorted_indexes = sorted_indexes[1:][keep_mask]

        # Return the proposals that are kept after NMS
        return proposals[torch.tensor(keep, dtype=torch.long)]

    def calculate_iou(self, proposal: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Intersection over Union (IoU) between a given proposal and a set of proposals.

        Args:
            proposal (torch.Tensor): A single proposal bounding box with shape (1, 4).
            proposals (torch.Tensor): A set of proposal bounding boxes with shape (num_proposals, 4).

        Returns:
            torch.Tensor: A tensor containing the IoU values with shape (num_proposals,).
        """
        # Calculate intersection and union areas
        inter_area, union_area = self._calculate_areas(proposal, proposals)

        # Calculate IoU
        iou = inter_area / union_area
        return iou

    def _calculate_areas(self, proposal: torch.Tensor, proposals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the intersection and union areas between a given proposal and a set of proposals.

        Args:
            proposal (torch.Tensor): A single proposal bounding box with shape (1, 4).
            proposals (torch.Tensor): A set of proposal bounding boxes with shape (num_proposals, 4).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the intersection area tensor with shape (num_proposals,)
                                                and the union area tensor with shape (num_proposals,).
        """
        x1, y1, x2, y2 = proposal[..., 0], proposal[..., 1], proposal[..., 2], proposal[..., 3]
        x1s, y1s, x2s, y2s = proposals[..., 0], proposals[..., 1], proposals[..., 2], proposals[..., 3]

        # Calculate intersection coordinates
        xi1 = torch.maximum(x1, x1s)
        yi1 = torch.maximum(y1, y1s)
        xi2 = torch.minimum(x2, x2s)
        yi2 = torch.minimum(y2, y2s)

        # Calculate intersection area
        inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)

        # Calculate union area
        proposal_area = (x2 - x1) * (y2 - y1)
        proposals_area = (x2s - x1s) * (y2s - y1s)
        union_area = proposal_area + proposals_area - inter_area

        return inter_area, union_area

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

        self.roi = ROIPooling((7,7), 1.0)

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
        batch_index = torch.cat([torch.full((len(batch), 1), i) for i, batch in enumerate(filtered_anchors)], dim = 0)
        roi = torch.cat([roi, batch_index], dim = 1)
        
        output = self.roi(feature_map, roi)
        
        return output