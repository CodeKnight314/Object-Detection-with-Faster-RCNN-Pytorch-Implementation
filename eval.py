import torch 
import torch.nn as nn 
from typing import List
from utils.box_utils import calculate_iou_batch

def calculate_precision(iou_matrix : torch.Tensor, 
                        positive_iou_threshold : float = 0.7, 
                        negative_iou_threshold : float = 0.3,): 
    
    """
    Calculating precision for a given set of proposals and references

    Args: 
        iou_matrix (torch.Tensor): An IOU matrix of shape (batch, number of propsoals, number of references)
        positive_iou_threshold (float): A float defining the positive IOU threshold for proposals and references
        negative_iou_threshold (float): A float defining the negative IOU threshold for proposals and references

    Returns: 
        float: returns a float based on precision formula (TP/(TP+NP)) or 0.0 if TP+NP = 0
    """
    
    max_iou, _ = iou_matrix.max(dim = 2)

    true_positive = torch.sum(max_iou >= positive_iou_threshold)
    false_positive = torch.sum(max_iou <= negative_iou_threshold)

    precision = true_positive.float() / (true_positive + false_positive).float() if true_positive + false_positive > 0 else 0.0

    return precision

def calculate_recall(iou_matrix : torch.Tensor, 
                     positive_iou_threshold : float = 0.5): 
    """
    """

    max_iou, _ = iou_matrix.max(dim = 2)

    true_positive = torch.sum(max_iou >= positive_iou_threshold)
    false_negative = torch.sum(max_iou <= positive_iou_threshold)

    recall = true_positive.float() / (true_positive + false_negative).float() if true_positive + false_negative > 0 else 0.0 

    return recall