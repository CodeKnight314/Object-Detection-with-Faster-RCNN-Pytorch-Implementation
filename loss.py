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
            List[torch.Tensor]: Maximum IOU value of each proposal against all ground truth box. ()
            List[torch.Tensor]: Maximum IOU index of each proposal, indicating which gt box has 
                          the highest IOU value with the respective proposal.
        """
        N, P, _ = proposals.shape


        iou_matrix = calculate_iou_batch(proposals=proposals, references=references)
        maxed_iou_matrix = []
        maxed_iou_index = []

        for item in iou_matrix: 
            max_iou, max_idx = item.max(dim=1)
            maxed_iou_matrix.append(max_iou)
            maxed_iou_index.append(max_idx)
        
        maxed_matrix = torch.stack(maxed_iou_matrix)

        labels = torch.full((N, P), -1, dtype=torch.float32, device=proposals.device)  # Ensure labels are float
        labels[maxed_matrix > self.positive_iou_anchor] = 1
        labels[maxed_matrix < self.negative_iou_anchor] = 0

        return labels, maxed_iou_matrix, maxed_iou_index
    
    def match_gt_to_box(self, positive_reference_index : List[torch.Tensor], references : List[torch.Tensor]):
        """
        Matches ground truth boundary box to each proposal at index batch_idx

        Args: 
            positive_reference_index (List[torch.Tensor]): a list of tensors, each item shaped as (number of positive proposals, ) 
                                                            with index relative to reference boxes' index for each batch
            references (List[torch.Tensor]): a list of tensors, each item shaped as (number of anchors, 4) with respective 
                                            coordinates for each reference item
        
        Returns: 
            List[torch.Tensor]: a list of tensors, each with shape (number of positive proposals, 4)
        """
        return [box[positive_reference_index[i]] for i, box in enumerate(references)] 

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
        labels, maxed_iou_matrix, maxed_iou_index = self.generate_labels(proposals=proposals, references=references)
        
        mask = labels != -1 
        BCELoss = F.binary_cross_entropy(cls_scores[:, :, 1], labels, reduction='none') * mask 
        BCELoss = BCELoss.sum() / mask.sum()

        positive_proposals = proposals[labels == 1]
        positive_iou_index = maxed_iou_index[labels == 1]
        pass