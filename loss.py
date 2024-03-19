import torch 
import torch.nn as nn 
from utils.box_utils import calculate_iou_batch, calculate_iou
from utils.box_utils import *
import torch.nn.functional as F
from typing import List
import torch.optim as opt 

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

    def forward(self, cls_scores: torch.Tensor, bbox_deltas: torch.Tensor, proposals: torch.Tensor, references: List[torch.Tensor]): 
        """
        Compute Losses for classification and bounding box regression.

        Args: 
            cls_scores (torch.Tensor): The predicted objectness score with shape (batch, number of proposals, 2)
            bbox_deltas (torch.Tensor): The predicted bounding box deltas with shape (batch, number of proposals, 4)
            proposals (torch.Tensor): The proposal anchors with shape (batch, number of proposals, 4). 
            references (List[torch.Tensor]): List of ground truth boxes for each image in the batch. 

        Returns: 
            torch.Tensor: The total loss combining objectness and bbox regression losses.
        """  
        labels, maxed_iou_matrix, maxed_iou_index = self.generate_labels(proposals=proposals, references=references)
        
        mask = labels != -1 
        BCELoss = F.binary_cross_entropy(cls_scores[:, :, 1], labels, reduction='none') * mask 
        objectness_loss = BCELoss.sum() / mask.sum()

        positive_mask = labels == 1
        if positive_mask.any():
            positive_proposals = proposals[positive_mask]
            positive_indices = torch.cat([idx[mask] for idx, mask in zip(maxed_iou_index, positive_mask)])
            matched_gt_boxes = torch.cat([refs[idx] for refs, idx in zip(references, positive_indices)])
            
            regression_targets = bbox_decode(positive_proposals, matched_gt_boxes)

            positive_deltas = bbox_deltas[positive_mask]

            bbox_loss = F.smooth_l1_loss(positive_deltas, regression_targets, reduction='mean')
        else:
            bbox_loss = 0.0

        total_loss = objectness_loss + bbox_loss
        
        return total_loss
    
def get_optimizer(model, lr : float, betas : Tuple[float], weight_decay : float): 
    """
    Helper function for defining optimizer 

    Args: 
        model : the model associated with the given optimizer 
        lr (float): learning rate for the optimizer 
        betas (Tuple[float]): a pair of floats
        weight_decay (float): determine rate of weight decay

    Returns:
        torch.optim : optimizer with the given parameters
    """
    return opt.Adam(model.parameters(), lr = lr, betas=betas, weight_decay=weight_decay)

def get_scheduler(optimizer : torch.optim, step_size : int, gamma : float): 
    """
    Helper function for defining learning rate scheduler -> may try to define my own for fun but who knows?

    Args: 
        optimizer (torch.optim): optimizer associated with the given learning rate scheduler 
        step_size (int): length of interval between each learning rate reduction 
        gamme (float): the rate at which the optimizer's learning rate decreases. New learning rate = lr * gamma at each step size interval
    """
    return opt.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)