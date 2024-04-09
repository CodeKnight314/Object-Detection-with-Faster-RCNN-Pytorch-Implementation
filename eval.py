import torch 
import torch.nn as nn 
from typing import List
from utils.box_utils import calculate_iou_batch

def calculate_precision(proposals : torch.Tensor, 
                        references : List[torch.Tensor],
                        positive_iou_threshold : int = 0.7, 
                        negative_iou_threshold : int = 0.3,): 
    
    iou_matrix = calculate_iou_batch(proposals=proposals, references=references)

    max_iou, _ = iou_matrix.max(dim = 2)

    true_positive = torch.sum(max_iou >= positive_iou_threshold)
    false_positive = torch.sum(max_iou <= negative_iou_threshold)

    precision = true_positive.float() / (true_positive + false_positive).float() if true_positive + false_positive > 0 else 0.0

    return precision
