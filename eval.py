import torch 
import torch.nn as nn 
from typing import List
from utils.box_utils import calculate_iou_batch

def calculate_precision(iou_matrix : torch.Tensor, 
                        positive_iou_threshold : float = 0.7, 
                        negative_iou_threshold : float = 0.3,): 
    
    """
    Calculating precision for a given IOU Matrix

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
    Calculating recall for a given IOU Matrix 

    Args: 
        iou_matrix (torch.Tensor): An IOU Matrix (batch, number of proposals, number of references)
        positive_iou_threshold (float): A float defining the positive IOU threshold for a boundary box to be true positive.
                                        All boundary boxes below the positive iou threshold are considered false negative.

    Returns: 
        float: returns a float based on recall formula (TP/(TP+FN)) or 0.0 if TP + FN = 0
    """

    max_iou, _ = iou_matrix.max(dim = 2)

    true_positive = torch.sum(max_iou >= positive_iou_threshold)
    false_negative = torch.sum(max_iou <= positive_iou_threshold)

    recall = true_positive.float() / (true_positive + false_negative).float() if true_positive + false_negative > 0 else 0.0 

    return recall

def calculate_f1_score(precision : float,
                       recall : float):
    
    """
    Calculating F1-Score based on precision and recall 

    Args: 
        precision (float): precision score of object detection model 
        recall (float)L: recall score of object detection model

    Returns: 
        float: returns a float based on the f1-score formula 2*(precision*recall)/(precision + recall)
    """

    return (2*precision*recall).float() / (precision + recall).float()