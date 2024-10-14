import torch 
import torch.nn as nn 
from loss import get_loss_functions, RPNLoss, FasterRCNNLoss
from dataset import get_dataset
from utils.log_writer import LOGWRITER
from Faster_RCNN import Faster_RCNN
from torch.utils.data import DataLoader
from Faster_RCNN import get_model
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
import os

def train_step(model: Faster_RCNN, 
               data: Tuple[torch.Tensor, Dict], 
               optimizer: torch.optim.Optimizer, 
               rpn_loss_function: RPNLoss, 
               frcnn_loss_function: FasterRCNNLoss): 
    """
    Training Step for training the model at each step. 

    Args: 
        model (Faster_RCNN): Faster RCNN Model for evaluation. 
        data (Tuple[torch.Tensor, Dict]): A tuple containing a batched tensor (N, C, H, W) and a dictionary with corresponding labels and bboxes.
        optimizer (torch.optim.Optimizer): Optimizer for model to update and backpropagate loss.
        rpn_loss_function (RPNLoss): RPN Loss function for calculating loss.
        frcnn_loss_function (FasterRCNNLoss): FRCNN Loss function for calculating loss.

    Returns: 
        Tuple[float, float]: The RPN and FRCNN losses as floats.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, gts = data
    images = images.to(device)
    bboxes = [item["boxes"].to(device) for item in gts]
    labels = [item["labels"].to(device) for item in gts]

    optimizer.zero_grad()

    frcnn_labels, frcnn_bboxes, rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors = model(images)

    rpn_total_loss, _, _ = rpn_loss_function(rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors, bboxes)

    frcnn_total_loss, _, _ = frcnn_loss_function(frcnn_labels, frcnn_bboxes, labels, bboxes)

    total_loss = rpn_total_loss + frcnn_total_loss

    if torch.isnan(total_loss):
        print("NaN encountered in loss. Skipping update.")
        return rpn_total_loss.item(), frcnn_total_loss.item()

    total_loss.backward()
    optimizer.step()

    return rpn_total_loss.item(), frcnn_total_loss.item()

def train(model: Faster_RCNN, 
          dataset: DataLoader, 
          logger: LOGWRITER, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler.StepLR,
          rpn_loss_function: RPNLoss, 
          frcnn_loss_function: FasterRCNNLoss, 
          epochs: int, 
          output_path: str, 
          weights: str): 
    """
    Training Protocol for a given model and dataset. 

    Args: 
        model (Faster_RCNN): Faster RCNN model for evaluation. Model path is loaded if valid. 
        dataset (DataLoader): Train dataset under the subclass of ObjectDetectionDataset. 
        logger (LOGWRITER): Log writer that takes kwargs and writes them to a txt file.
        optimizer (torch.optim.Optimizer): Optimizer for model to update and propagate loss.
        scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler to update the learning rate of the optimizer.
        rpn_loss_function (RPNLoss): RPN Loss function for calculating loss.
        frcnn_loss_function (FasterRCNNLoss): FRCNN Loss function for calculating loss.
        epochs (int): Total number of epochs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    
    if weights:
        model.load_state_dict(torch.load(weights, weights_only=True,
                                         map_location=device))

    best_loss = float('inf')

    for epoch in range(epochs):
        batched_values = []

        for data in tqdm(dataset, desc=f"[{epoch+1}/{epochs}] Training"):
            values = train_step(model, data, optimizer, rpn_loss_function, frcnn_loss_function)
            batched_values.append(values)

        batched_values = torch.tensor(batched_values, dtype=torch.float32)
        averaged_values = batched_values.mean(dim=0)

        logger.write(epoch + 1, 
                     RPN_Loss=averaged_values[0].item(), 
                     FRCNN_Loss=averaged_values[1].item())

        if best_loss > (averaged_values[0] + averaged_values[1]):
            if not os.path.exists(output_path):
                os.makedirs(output_path) 
            torch.save(model.state_dict(), os.path.join(output_path, f"FRCNN_model_{epoch+1}.pth"))
            best_loss = averaged_values[0] + averaged_values[1]

        scheduler.step()

def main(args): 
    train_dl = get_dataset(args.root_dir, args.img_h, args.img_w, "train", args.batch_size)
    
    model = get_model(cls_count=2, training=True)  # cls_count=2 since it's either human or not human

    logger = LOGWRITER(args.output_dir, args.epochs)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, eta_min=args.lr / 100, T_max=args.epochs * 0.75)

    rpn_loss, frcnn_loss = get_loss_functions((0.7, 0.3))

    train(model=model, 
          dataset=train_dl, 
          logger=logger, 
          optimizer=opt,
          scheduler=scheduler, 
          rpn_loss_function=rpn_loss, 
          frcnn_loss_function=frcnn_loss, 
          epochs=args.epochs,
          output_path=args.outpath, 
          weights=args.weights)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--weights", type=str)
    
    args = parser.parse_args()
    
    main(args)