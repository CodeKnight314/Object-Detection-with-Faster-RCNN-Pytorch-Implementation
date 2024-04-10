import torch 
import torch.nn as nn 
from typing import List
from utils.box_utils import calculate_iou_batch

def calculate_precision(proposals : torch.Tensor, 
                        references : List[torch.Tensor],
                        positive_iou_threshold : float = 0.7, 
                        negative_iou_threshold : float = 0.3,): 
    
    """
    Calculating precision for a given set of proposals and references

    Args: 
        proposals (torch.Tensor): A batch of proposals with shape (batch, number of proposals, 4)
        references (List[torch.Tensor]): A list of of reference boundary boxes (number of references, 4), with length batch
        positive_iou_threshold (float): A float defining the positive IOU threshold for proposals and references
        negative_iou_threshold (float): A float defining the negative IOU threshold for proposals and references

    Returns: 
        float: returns a float based on precision formula (TP/(TP+NP)) or 0.0 if TP+NP = 0
    """
    
    iou_matrix = calculate_iou_batch(proposals=proposals, references=references)

    max_iou, _ = iou_matrix.max(dim = 2)

    true_positive = torch.sum(max_iou >= positive_iou_threshold)
    false_positive = torch.sum(max_iou <= negative_iou_threshold)

    precision = true_positive.float() / (true_positive + false_positive).float() if true_positive + false_positive > 0 else 0.0

    return precision
