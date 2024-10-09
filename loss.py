import torch 
import torch.nn as nn 
from utils.box_utils import calculate_iou_batch, calculate_iou
from utils.box_utils import *
import torch.nn.functional as F
from typing import List
import torch.optim as opt 
import random 
import time

class FasterRCNNLoss(nn.Module):
    def __init__(self, positive_iou_threshold: float, negative_iou_threshold: float):
        super(FasterRCNNLoss, self).__init__()
        self.positive_iou_anchor = positive_iou_threshold
        self.negative_iou_anchor = negative_iou_threshold

    def forward(self, frcnn_cls: torch.Tensor, frcnn_bbox: torch.Tensor, frcnn_labels: List[torch.Tensor], frcnn_gt_bbox: List[torch.Tensor]):
        """
        Calculate the classification and regression losses for Faster R-CNN.
        Args:
            frcnn_cls (torch.Tensor): Predicted class scores with shape (batch_size, num_proposals, num_classes).
            frcnn_bbox (torch.Tensor): Predicted bounding box deltas with shape (batch_size, num_proposals, num_classes * 4).
            frcnn_labels (List[torch.Tensor]): List of ground truth class labels for each image in the batch.
            frcnn_gt_bbox (List[torch.Tensor]): List of ground truth boxes for each image in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, regression loss, and classification loss.
        """
        batch_size = frcnn_cls.size(0)
        classification_loss = 0.0
        regression_loss = 0.0

        for i in range(batch_size):
            # Get ground truth labels and bounding boxes for current batch
            gt_labels = frcnn_labels[i]
            gt_boxes = frcnn_gt_bbox[i]

            # Calculate IoU between predicted boxes and ground truth boxes
            iou_matrix = calculate_iou(frcnn_bbox[i], gt_boxes)

            # Determine positive and negative samples
            max_iou, max_indices = iou_matrix.max(dim=1)
            positive_indices = torch.where(max_iou >= self.positive_iou_anchor)[0]
            negative_indices = torch.where(max_iou < self.negative_iou_anchor)[0]

            # Classification Loss
            labels = torch.full((frcnn_cls.size(1),), -1, dtype=torch.long, device=frcnn_cls.device)
            labels[positive_indices] = gt_labels[max_indices[positive_indices]]
            labels[negative_indices] = 0  # Background class

            valid_indices = labels != -1
            if valid_indices.sum() > 0:
                classification_loss += F.cross_entropy(frcnn_cls[i][valid_indices], labels[valid_indices])

            # Regression Loss
            if positive_indices.numel() > 0:
                pos_predicted_boxes = frcnn_bbox[i][positive_indices]
                pos_gt_boxes = gt_boxes[max_indices[positive_indices]]
                regression_loss += F.smooth_l1_loss(pos_predicted_boxes, pos_gt_boxes, reduction='mean')

        total_loss = classification_loss + regression_loss
        return total_loss, regression_loss, classification_loss
            
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
            torch.Tensor: Labels for each proposal in the batch. (batch_number, number of proposals)
            List[torch.Tensor]: Maximum IOU value of each proposal against all ground truth box.
            List[torch.Tensor]: Maximum IOU index of each proposal, indicating which gt box has 
                                the highest IOU value with the respective proposal.
        """
        N, P, _ = proposals.shape
        
        # Initialize labels and IOU storage
        labels = torch.full((N, P), -1, dtype=torch.int64, device=proposals.device)
        maxed_iou_matrix = []
        maxed_iou_index = []
        
        for i, ref in enumerate(references):
            if ref.numel() == 0:
                maxed_iou_matrix.append(torch.zeros(P, device=proposals.device))
                maxed_iou_index.append(torch.zeros(P, dtype=torch.int64, device=proposals.device))
                continue
            
            iou = calculate_iou(proposals[i], ref)
            max_iou, max_idx = iou.max(dim=1)
            maxed_iou_matrix.append(max_iou)
            maxed_iou_index.append(max_idx)
        
            labels[i][max_iou > self.positive_iou_anchor] = 1
            labels[i][max_iou < self.negative_iou_anchor] = 0
        
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
        matched_boxes = []
        for i, box in enumerate(references):
            if len(box) == 0 or len(positive_reference_index[i]) == 0:
                continue
            valid_indices = positive_reference_index[i][positive_reference_index[i] < box.size(0)]
            matched_boxes.append(box[valid_indices])
    
        return matched_boxes

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
        labels, maxed_iou_matrix, maxed_iou_index = self.generate_labels(proposals, references)
        
        mask = labels != -1
        
        if mask.sum() > 0:
            BCELoss = F.binary_cross_entropy_with_logits(cls_scores[:, :, 1], labels.float(), reduction='none') * mask
            objectness_loss = BCELoss.sum() / mask.sum()
        else:
            objectness_loss = torch.tensor(0.0, device=cls_scores.device)
        
        bbox_loss = 0.0
        
        positive_mask = labels == 1
        if positive_mask.sum() > 0:
            positive_proposals = proposals[positive_mask]
            
            positive_indices = [idx[mask] for idx, mask in zip(maxed_iou_index, positive_mask)]
            
            matched_gt_boxes = self.match_gt_to_box(positive_indices, references)
        
            if len(matched_gt_boxes) > 0 and any([len(boxes) > 0 for boxes in matched_gt_boxes]):
                matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
                
                regression_targets = bbox_decode(positive_proposals, matched_gt_boxes)
                
                positive_deltas = bbox_deltas[positive_mask]
                
                bbox_loss = F.smooth_l1_loss(positive_deltas, regression_targets, reduction='mean')
        
        total_loss = objectness_loss + bbox_loss
        
        return total_loss, objectness_loss, bbox_loss

def get_loss_functions(iou_thresholds : Tuple[int, int]):
    """
    Helper function for defining RPN & Faster RCNN loss calculation class 

    Args: 
        iou_thresholds (Tuple[int]): Contains two iou threshold values, one for positive and negative, respectively.
    
    Returns: 
        RPNLoss : loss calculation class for RPN outputs 
        FasterRCNNLoss : loss calculation class for Faster RCNN
    """
    assert len(iou_thresholds) == 2,f"[Error] Expected 2 iou threshold values but received {len(iou_thresholds)}"
    positive_iou_threshold, negative_iou_threshold = iou_thresholds

    rpn_loss = RPNLoss(positive_iou_threshold=positive_iou_threshold, negative_iou_threshold=negative_iou_threshold)
    frcnn_loss = FasterRCNNLoss(positive_iou_threshold=positive_iou_threshold, negative_iou_threshold=negative_iou_threshold)

    return rpn_loss, frcnn_loss