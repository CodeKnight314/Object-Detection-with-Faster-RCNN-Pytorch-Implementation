import torch 
import configs
import torch.nn as nn 
from loss import *
from torch_snippets import *
from dataset import *
from utils.log_writer import LOGWRITER
from Faster_RCNN import Faster_RCNN
from torch.utils.data import DataLoader
from Faster_RCNN import *
from tqdm import tqdm
from typing import Dict

def train_step(model : Faster_RCNN, 
               data : Tuple[torch.Tensor, Dict], 
               optimizer : torch.optim, 
               rpn_loss_function : RPNLoss, 
               frcnn_loss_function : FasterRCNNLoss): 
    """
    Training Step for training the model at each step. 

    Args: 
        model (Faster_RCNN): Faster RCNN Model for evaluation. 
        data (Tuple[torch.Tensor, DIct]): A tuple containing a batched tensor (N, C, H, W) and a dictionary with corresponding labels and bboxes
        optimizer (torch.optim): Optimizer for model to update and backpropagate loss.
        rpn_loss_function (RPNLoss): RPN Loss function for calculating loss.
        frcnn_loss_function (FasterRCNNLoss): FRCNN Loss function for calculating loss.

    Returns: 
        rpn_total_loss.item() (float): rpn_loss as a float 
        frcnn_total_loss.item() (float): frcnn_loss as a float 
        rpn_runtime (float): RPN runtime in seconds 
        frcnn_runtime (float): FRCNN runtime in seconds 
        model.time_records["Total"] (float): model runtime in seconds
    """
    images, gts = data
    bboxes = [item["boxes"] for item in gts]
    labels = [item["labels"] for item in gts]

    optimizer.zero_grad()

    frcnn_labels, frcnn_bboxes, rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors = model(images)

    rpn_start = time.time()
    rpn_total_loss, _, _ = rpn_loss_function(rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors, bboxes)
    rpn_runtime = time.time() - rpn_start

    frcnn_start = time.time()
    frcnn_total_loss, _, _ = frcnn_loss_function(frcnn_labels, frcnn_bboxes, labels, bboxes)
    frcnn_runtime = time.time() - frcnn_start

    total_loss = rpn_total_loss + frcnn_total_loss
    total_loss.backward()
    optimizer.step()

    return rpn_total_loss.item(), frcnn_total_loss.item(), rpn_runtime, frcnn_runtime, model.time_records["Total"]


def train(model : Faster_RCNN, 
          dataset : DataLoader, 
          logger : LOGWRITER, 
          optimizer : torch.optim, 
          scheduler : opt.lr_scheduler.StepLR,
          rpn_loss_function : RPNLoss, 
          frcnn_loss_function : FasterRCNNLoss, 
          epochs : int): 
    """
    Training Protocol for a given model and dataset. 

    Args: 
        model (Faster_RCNN): Faster RCNN model for evaluation. Model path is loaded if valid. 
        dataset (DataLoader): Train dataset under the subclass of ObjectDetectionDataset. 
        logger (LOGWRITER): Log writer that takes kwargs and writes them to txt file.
        optimizer (torch.optim): Optimizer for model to update and propagate loss.
        scheduler (opt.lr_scheduler.StepLR): Learning rate scheduler to update the learning rate of the optimizer
        rpn_loss_function (RPNLoss): RPN Loss function for calculating loss 
        frcnn_loss_function (FasterRCNNLoss): FRCNN Loss function for calculating loss 
        epoch (int): total number of epochs
    """
    model.train()
    if configs.model_path:
        model.load_state_dict(torch.load(configs.model_path, 
                                         map_location="cuda" if torch.cuda.is_available() else "cpu"))

    best_loss = float('inf')

    for epoch in range(epochs):

        batched_values = []

        for data in tqdm(dataset, desc = f"[{epoch+1}/{epochs}] Training"):
            values = train_step(model, data, optimizer, rpn_loss_function, frcnn_loss_function)
            batched_values.append(values)

        averaged_values = torch.sum(torch.tensor(batched_values), dim = 1) / len(batched_values)

        logger.write(epoch, RPN_Loss = averaged_values[0], 
                     FRCNN_Loss = averaged_values[1], 
                     RPN_Runtime = averaged_values[2], 
                     FRCNN_Runtime = averaged_values[3],
                     Model_Runtime = averaged_values[4])

        if best_loss > (averaged_values[0] + averaged_values[1]):
            if not os.path.exists(configs.model_save_path):
                os.makedirs(configs.model_save_path) 
            torch.save(model.state_dict(), os.path.join(configs.model_save_path, f"FRCNN_model_{epoch+1}.pth"))
            best_loss = averaged_values[0] + averaged_values[1]

        scheduler.step()

def main(): 
    train_dataset = load_COCO_dataset(configs.root_dir,
                          configs.image_height, 
                          configs.image_width, 
                          configs.annotation_dir, 
                          transforms=None, 
                          mode="train"),
    
    train_dl = load_dataloaders(train_dataset[0], 
                                configs.batch_number,
                                shuffle = True, 
                                drop_last = True)

    print("[INFO] Dataloader loaded successfully")
    print(f"[INFO] total training samples {len(train_dataset[0])}")
    
    model = get_model(cls_count = len(train_dataset[0].id_to_category), training=True)

    logger = LOGWRITER(configs.output_dir, configs.epochs)

    opt = get_optimizer(model, lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)

    scheduler = get_scheduler(optimizer = opt, step_size = configs.epochs//5, gamma = 0.5)

    rpn_loss, frcnn_loss = get_loss_functions((0.7, 0.3))

    train(model=model, 
          dataset=train_dl, 
          logger=logger, 
          optimizer=opt,
          scheduler=scheduler, 
          rpn_loss_function=rpn_loss, 
          frcnn_loss_function=frcnn_loss, 
          epochs=configs.epochs)

if __name__ == "__main__": 
    main()