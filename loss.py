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

    def select_frcnn_bbox_from_cls(self, frcnn_cls: torch.Tensor, frcnn_bbox: torch.Tensor):
        _, prediction = frcnn_cls.max(dim=2)
        prediction_index = prediction[..., None, None].expand(-1, -1, 1, 4)
        selected_bbox = torch.gather(frcnn_bbox, dim=2, index=prediction_index).squeeze(2)
        return prediction, selected_bbox

    def forward(self, frcnn_cls: torch.Tensor, frcnn_bbox: torch.Tensor, frcnn_labels: List[torch.Tensor], frcnn_gt_bbox: List[torch.Tensor]):

        frcnn_pred, frcnn_bbox = self.select_frcnn_bbox_from_cls(frcnn_cls, frcnn_bbox)

        highest_iou = [matrix.max(dim = 1)[1].unsqueeze(0) for matrix in calculate_iou_batch(frcnn_bbox, frcnn_gt_bbox)]

        gts_cls = [] 
        gts_bbox = []
        for i in range(frcnn_cls.size(0)): 
            gts_cls.append(frcnn_labels[i][highest_iou[i]].squeeze(0))
            gts_bbox.append(frcnn_gt_bbox[i][highest_iou[i]].squeeze(0))

        gts_cls = torch.stack(gts_cls).to(frcnn_pred.device).long()
        gts_bbox = torch.stack(gts_bbox).to(frcnn_pred.device)

        valid_indices = (gts_cls != -1).view(-1)

        if valid_indices.any():
            frcnn_cls = frcnn_cls.view(-1, frcnn_cls.size(-1))[valid_indices]
            gts_cls = gts_cls.view(-1)[valid_indices]
        else:
            classification_loss = torch.tensor(0.0).to(frcnn_cls.device)  # Example fallback

        
        classification_loss = F.cross_entropy(frcnn_cls, gts_cls)        

        regression_loss = F.smooth_l1_loss(frcnn_bbox, gts_bbox)
        
        total_loss = regression_loss + classification_loss 

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
        
        BCELoss = F.binary_cross_entropy_with_logits(cls_scores[:, :, 1], labels.float(), reduction='none') * mask
        objectness_loss = BCELoss.sum() / mask.sum()
        
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

def get_loss_functions(iou_thresholds : Tuple[int, int]):
    """
    Helper function for defining RPN & Faster RCNN loss calculation class 

    Args: 
        iou_thresholds (Tuple[int]): Contains two iou threshold values, one for positive and negative, respectively.
    
    Returns: 
        RPNLoss : loss calculation class for RPN outputs 
        FasterRCNNLoss : loss calculation class for Faster RCNN
    """
    assert len(iou_thresholds) == 2,f"[Error] Expected 2 iou threshold values but receievd {len(iou_thresholds)}"
    positive_iou_threshold, negative_iou_threshold = iou_thresholds

    rpn_loss = RPNLoss(positive_iou_threshold=positive_iou_threshold, negative_iou_threshold=negative_iou_threshold)
    frcnn_loss = FasterRCNNLoss(positive_iou_threshold=positive_iou_threshold, negative_iou_threshold=negative_iou_threshold)

    return rpn_loss, frcnn_loss

def main(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_idx = 16 
    num_of_proposals = 300 
    num_of_classes = 21

    rpn_loss, frcnn_loss = get_loss_functions((0.7, 0.3)) 

    start = time.time()
    rpn_cls = torch.rand((batch_idx, num_of_proposals, 2), dtype = torch.float32, device = device) 

    rpn_bbox = torch.rand((batch_idx, num_of_proposals, 4), dtype = torch.float32, device = device) * 32

    anchors = torch.rand((batch_idx, num_of_proposals, 4), dtype = torch.float32, device = device)

    references = [torch.rand((random.randint(1, 10), 4), dtype = torch.float32, device = device) for _ in range(batch_idx)]

    total_loss_rpn = rpn_loss(rpn_cls, rpn_bbox, anchors, references)

    print(f"RPN Loss: {total_loss_rpn.item()}")

    frcnn_cls = torch.rand((batch_idx, num_of_proposals, num_of_classes), dtype = torch.float32, device = device) 

    frcnn_bbox = torch.rand((batch_idx, num_of_proposals, num_of_classes * 4), dtype = torch.float32, device = device)

    frcnn_labels = [torch.rand((10, 1), dtype = torch.float32, device = device) for _ in range(batch_idx)]

    frcnn_gt_bbox= [torch.rand((10, 4), dtype = torch.float32, device = device) for _ in range(batch_idx)]

    total_loss, avg_classification_loss, avg_regression_loss = frcnn_loss(frcnn_cls, frcnn_bbox, frcnn_labels, frcnn_gt_bbox)

    end = time.time() - start
    print(f"FRCNN Loss: {total_loss.item()}")

    print(f"Total Runtime: {end}")

if __name__ == "__main__": 
    main()
