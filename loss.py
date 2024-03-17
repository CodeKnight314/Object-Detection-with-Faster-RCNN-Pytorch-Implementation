import torch 
import torch.nn as nn 
from utils.box_utils import calculate_iou_batch, calculate_iou
import torch.nn.functional as F
from typing import List

class FasterRCNNLoss(nn.Module): 
    def __init__(self): 
        pass 

    def __compute_Loss__(self): 
        pass 

    def forward(self, x): 
        pass

class RPNLoss(nn.Module): 
    def __init__(self, positive_iou_threshold=0.7, negative_iou_threshold=0.3):
        super(RPNLoss, self).__init__()
        self.positive_iou_anchor = positive_iou_threshold
        self.negative_iou_anchor = negative_iou_threshold

    def generate_labels(self, proposals: torch.Tensor, references: List[torch.Tensor]):
        """
        Generate labels for proposals based on IoU with ground truth boxes.

        Args:
            proposals (torch.Tensor): The proposal anchors with shape (batch, number of proposals, 4).
            references (List[torch.Tensor]): List of ground truth boxes for each image in the batch.
        
        Returns:
            torch.Tensor: Labels for each proposal in the batch.
        """
        N, P, _ = proposals.shape
        iou_matrix = torch.stack([matrix.max(dim=1)[0] for matrix in calculate_iou_batch(proposals=proposals, references=references)])

        labels = torch.full((N, P), -1, dtype=torch.float32, device=proposals.device)  # Ensure labels are float
        labels[iou_matrix > self.positive_iou_anchor] = 1
        labels[iou_matrix < self.negative_iou_anchor] = 0

        return labels

    def forward(self, cls_scores: torch.Tensor, proposals: torch.Tensor, references: List[torch.Tensor]): 
        """
        Compute Losses for classification and boundary box regression

        Args: 
            cls_scores (torch.Tensor): The predicted objectness score with shape (batch, number of proposals, 2)
            proposals (torch.Tensor): The proposal anchors with shape (batch, number of proposals, 4). 
            references (List[torch.Tensor]): List of ground truth boxes for each image in the batch. 
                                             Each ground truth box is formatted as (number of references, 4)
        
        Returns: 

        """  
        labels = self.generate_labels(proposals=proposals, references=references)
        BCELoss = F.binary_cross_entropy(cls_scores[:, :, 1], labels, reduction='mean')

        pass