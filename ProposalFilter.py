import torch 
import torch.nn as nn 
from typing import Tuple

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
