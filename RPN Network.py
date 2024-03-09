import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Tuple

# RPN Head -> Classify Objectness Score and Bbox Regression
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

class AnchorGenerator(nn.Module):
    """
    Generates anchor boxes for a given set of sizes and aspect ratios.

    Attributes:
        sizes (Tuple[int]): A tuple of integers representing the sizes of the anchor boxes.
        aspect_ratios (Tuple[float]): A tuple of floats representing the aspect ratios of the anchor boxes.
    """
    def __init__(self, sizes: Tuple[int], aspect_ratios: Tuple[float]) -> None:
        super(AnchorGenerator, self).__init__()
        assert len(sizes) == len(aspect_ratios), f"[Error] AnchorGenerator tuple sizes do not match. sizes : {len(sizes)}, aspect ratios: {len(aspect_ratios)}"
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = self.generate_cell_anchors(self.sizes, self.aspect_ratios)

    def generate_cell_anchors(self, sizes: Tuple[int], aspect_ratios: Tuple[float], dtype: torch.dtype = torch.float32, device="cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
        """
        Generates anchor boxes for each cell in the feature map.

        Args:
            sizes (Tuple[int]): A tuple of integers representing the sizes of the anchor boxes.
            aspect_ratios (Tuple[float]): A tuple of floats representing the aspect ratios of the anchor boxes.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float32.
            device (str, optional): Device where the tensors will be stored. Defaults to "cuda" if available, else "cpu".

        Returns:
            torch.Tensor: A tensor containing the generated anchor boxes with shape (num_anchors, 4).
        """
        size_tensors = torch.tensor(sizes, dtype=dtype, device=device)
        ar_tensors = torch.tensor(aspect_ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ar_tensors)
        w_ratios = 1 / h_ratios

        hs = (h_ratios[:, None] * size_tensors[None, :]).view(-1)
        ws = (w_ratios[:, None] * size_tensors[None, :]).view(-1)

        cell_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return cell_anchors.round()

    def number_of_anchors_per_location(self) -> int:
        """
        Returns the number of anchors per location.

        Returns:
            int: The number of anchors per location.
        """
        return len(self.sizes) * len(self.aspect_ratios)

    def forward(self, images: torch.Tensor, feature_maps: torch.Tensor) -> torch.Tensor:
        """
        Generates anchors for the entire image.

        Args:
            images (torch.Tensor): The input images with shape (batch_size, channels, height, width).
            feature_maps (torch.Tensor): The feature maps from the backbone network with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor containing the generated anchors for each feature map with shape (batch_size, num_anchors, 4).
        """
        image_dim = [(images.size(2), images.size(3)) for _ in range(images.size(0))]
        feature_dim = [(feature_maps.size(2), feature_maps.size(3)) for _ in range(feature_maps.size(0))]
        strides = [(img_dim[0] / f_dim[0], img_dim[1] / f_dim[1]) for img_dim, f_dim in zip(image_dim, feature_dim)]
        dtype, device = feature_maps.dtype, feature_maps.device

        anchors = torch.stack([self.generate_image_anchors(f_dim, stride, feature_maps.dtype, feature_maps.device) for f_dim, stride in zip(feature_dim, strides)], dim=0)
        return anchors

    def generate_image_anchors(self, feature_map_size: Tuple[int, int], stride: Tuple[float, float], dtype: torch.dtype, device: str) -> torch.Tensor:
        """
        Generates anchors for a single feature map.

        Args:
            feature_map_size (Tuple[int, int]): The size of the feature map (height, width).
            stride (Tuple[float, float]): The stride of the feature map relative to the original image size.
            dtype (torch.dtype): Data type of the tensors.
            device (str): Device where the tensors will be stored.

        Returns:
            torch.Tensor: A tensor containing the generated anchors for the feature map with shape (num_anchors, 4).
        """
        grid_height, grid_width = feature_map_size
        shifts_x = torch.arange(0, grid_width, dtype=dtype, device=device) * stride[0]
        shifts_y = torch.arange(0, grid_height, dtype=dtype, device=device) * stride[1]
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack((-shift_x, -shift_y, shift_x, shift_y), dim=-1).view(-1, 4)

        stride_tensor = torch.tensor(stride, dtype=dtype, device=device).view(1, 2).repeat(1, 2)
        scaled_anchors = self.cell_anchors * stride_tensor

        all_anchors = scaled_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        return all_anchors.view(-1, 4)

def bbox_encode(anchors: torch.Tensor, anchor_offsets: torch.Tensor) -> torch.Tensor:
    """
    Decodes the bounding box deltas and applies them to the anchor boxes.

    Args:
        anchors (torch.Tensor): The anchor boxes tensor with shape (batch_size, num_anchors, 4).
            Each anchor box is represented as (x_min, y_min, x_max, y_max).
        anchor_offsets (torch.Tensor): The predicted bounding box deltas with shape (batch_size, num_anchors, 4).
            Each delta is represented as (dx, dy, dw, dh).

    Returns:
        torch.Tensor: The adjusted anchor boxes tensor with shape (batch_size, num_anchors, 4).
            Each adjusted box is represented as (x_min, y_min, x_max, y_max).
    """
    # Calculate the widths and heights of the anchors
    anchor_widths = anchors[:, :, 2] - anchors[:, :, 0]
    anchor_heights = anchors[:, :, 3] - anchors[:, :, 1]

    # Calculate the center coordinates of the anchors
    anchor_center_x = anchors[:, :, 0] + 0.5 * anchor_widths
    anchor_center_y = anchors[:, :, 1] + 0.5 * anchor_heights

    # Apply the offsets to the anchor center coordinates
    new_center_x = anchor_offsets[:, :, 0] * anchor_widths + anchor_center_x
    new_center_y = anchor_offsets[:, :, 1] * anchor_heights + anchor_center_y

    # Apply the scaling factors to the widths and heights
    new_widths = torch.exp(anchor_offsets[:, :, 2]) * anchor_widths
    new_heights = torch.exp(anchor_offsets[:, :, 3]) * anchor_heights

    # Calculate the top-left and bottom-right coordinates of the adjusted anchors
    top_left_x = new_center_x - 0.5 * new_widths
    top_left_y = new_center_y - 0.5 * new_heights
    bottom_right_x = new_center_x + 0.5 * new_widths
    bottom_right_y = new_center_y + 0.5 * new_heights

    # Combine the coordinates into a single tensor
    adjusted_anchors = torch.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim=2)

    return adjusted_anchors

def calculate_iou(proposals: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """
    Calculate IOU for a set of proposals with a set of references.

    Args:
        proposals (torch.Tensor): The proposal anchors with shape (number of proposals, 4).
        references (torch.Tensor): The reference anchors with shape (number of references, 4).

    Returns:
        torch.Tensor: Tensor containing the IOU values with shape (number of proposals, number of references).
    """
    # Calculate the intersection
    max_xy = torch.min(proposals[:, None, 2:], references[:, 2:])
    min_xy = torch.max(proposals[:, None, :2], references[:, :2])
    intersection = torch.clamp(max_xy - min_xy, min=0)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    # Calculate the union
    proposals_area = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    references_area = (references[:, 2] - references[:, 0]) * (references[:, 3] - references[:, 1])
    union_area = proposals_area[:, None] + references_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

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
        return filtered_anchors