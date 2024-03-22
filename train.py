import torch 
import configs
import torch.nn as nn 
from tqdm import tqdm 
from loss import *
from torch_snippets import *
from utils.log_writer import LOGWRITER
from Faster_RCNN import Faster_RCNN
from torch.utils.data import DataLoader

def train(model : Faster_RCNN, dataset : DataLoader, logger : LOGWRITER, scheduler : opt.lr_scheduler.StepLR,
          rpn_loss_function : RPNLoss, frcnn_loss_function : FasterRCNNLoss, epochs : int): 
    
    if configs.model_path:
        model.load_state_dict(torch.load(configs.model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

    best_loss = float('inf')

    for epoch in range(epochs):

        rpn_loss = 0.0
        frcnn_loss = 0.0 

        for i, data in enumerate(tqdm(dataset, desc = f"[{epoch}/{epochs}] Training:")):
            images, gts = data 
            bboxes = gts["boxes"]
            labels = gts["labels"]

            frcnn_labels, frcnn_bboxes = model(images)


