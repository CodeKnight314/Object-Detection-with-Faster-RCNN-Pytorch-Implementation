import torch 
from utils.box_utils import calculate_iou_batch
from utils.log_writer import LOGWRITER
from Faster_RCNN import Faster_RCNN, get_model
from typing import Dict, Tuple
from loss import get_loss_functions, FasterRCNNLoss, RPNLoss
from dataset import get_dataset, DataLoader
import argparse
import os
from tqdm import tqdm

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

def eval_step(model : Faster_RCNN, 
              data : Tuple[torch.Tensor, Dict], 
              rpn_loss_function : RPNLoss, 
              frcnn_loss_function : FasterRCNNLoss):
    """
    Evaluation Step for evaluating the model at each step. 

    Args: 
        model (Faster_RCNN): Faster RCNN Model for evaluation. 
        data (Tuple[torch.Tensor, DIct]): a tuple containing a batched tensor (N, C, H, W) and a dictionary with corresponding labels and bboxes
        rpn_loss_function (RPNLoss): RPN Loss function for calculating loss.
        frcnn_loss_function (FasterRCNNLoss): FRCNN Loss function for calculating loss. 

    Return: 
        rpn_total_loss.item() (float): rpn_loss as a float 
        frcnn_total_loss.item() (float): frcnn_loss as a float 
        rpn_runtime (float): RPN runtime in seconds 
        frcnn_runtime (float): FRCNN runtime in seconds 
        model.time_records["Total"] (float): model runtime in seconds
        precision (float): precision score based on model outputs 
        recall (float): recall score based on model outputs 
        f1_score (float): f1_score based on model outputs
    """
    images, gts = data 
    bboxes = [item["boxes"] for item in gts]
    labels = [item["labels"] for item in gts]

    frcnn_labels, frcnn_bboxes, rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors = model(images)

    rpn_total_loss, _, _ = rpn_loss_function(rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors, bboxes)

    frcnn_total_loss, _, _ = frcnn_loss_function(frcnn_labels, frcnn_bboxes, labels, bboxes)

    iou_matrix = calculate_iou_batch(frcnn_bboxes, bboxes)

    precision = calculate_precision(iou_matrix=iou_matrix, positive_iou_threshold=0.7, negative_iou_threshold=0.3)
    recall = calculate_recall(iou_matrix=iou_matrix, positive_iou_threshold=0.5)
    f1_score = calculate_f1_score(precision=precision, recall=recall)

    return rpn_total_loss.item(), frcnn_total_loss.item(), precision, recall, f1_score

def eval(model : Faster_RCNN,
        dataset : DataLoader, 
        logger : LOGWRITER, 
        rpn_loss_function : RPNLoss, 
        frcnn_loss_function : FasterRCNNLoss,
        epochs : int,
        model_path : str): 
    """
    Evaluation Protocol for a given model and dataset 

    Args: 
        model (Faster_RCNN): Faster RCNN model for evaluation. Model path is loaded if valid. 
        dataset (DataLoader): Validation dataset under the subclass of ObjectDetectionDataset. 
        logger (LOGWRITER): Log writer that takes kwargs and writes them to txt file.
        rpn_loss_function (RPNLoss): RPN Loss function for calculating loss 
        frcnn_loss_function (FasterRCNNLoss): FRCNN Loss function for calculating loss 
        epoch (int): total number of epochs
    """
    model.eval()
    if os.path.exists(model_path): 
        model.load_state_dict(torch.load(model_path, map_location = "cuda" if torch.cuda.is_available() else "cpu"))
        
    for epoch in range(epochs):

        batched_values = []

        for data in tqdm(dataset, desc = f"[{epoch+1}/{epochs}]"):
            values = eval_step(model=model, data=data, rpn_loss_function=rpn_loss_function, frcnn_loss_function=frcnn_loss_function)
            batched_values.append(values)

        averaged_values = torch.sum(torch.tensor(batched_values), dim = 1) / len(batched_values)

        logger.write(epoch + 1, 
                     RPN_Loss = averaged_values[0], 
                     FRCNN_Loss = averaged_values[1],  
                     precision = averaged_values[2], 
                     recall = averaged_values[3], 
                     f1_score = averaged_values[4])
        
def main(args):
    eval_dl = get_dataset(args.root_dir, args.img_h, args.img_w, "val", args.batch_size)
    
    model = get_model(cls_count=2, training=True)  # cls_count=2 since it's either human or not human

    logger = LOGWRITER(args.output_dir, args.epochs)
    
    rpn_loss, frcnn_loss = get_loss_functions((0.7, 0.3))
    
    eval(model,
        eval_dl, 
        logger, 
        rpn_loss,
        frcnn_loss,
        args.epochs,
        args.model_path)
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for dataset")
    parser.add_argument("--img_h", type=int, default=800, help="Image height for resizing")
    parser.add_argument("--img_w", type=int, default=800, help="Image width for resizing")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation logs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of evaluation epochs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model for evaluation")

    args = parser.parse_args()
    main(args)