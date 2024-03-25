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

def train(model : Faster_RCNN, dataset : DataLoader, logger : LOGWRITER, optimizer : torch.optim, scheduler : opt.lr_scheduler.StepLR,
          rpn_loss_function : RPNLoss, frcnn_loss_function : FasterRCNNLoss, epochs : int): 
    
    if configs.model_path:
        model.load_state_dict(torch.load(configs.model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

    best_loss = float('inf')

    for epoch in range(epochs):

        rpn_loss = 0.0
        frcnn_loss = 0.0 

        for data in tqdm(dataset, desc = f"[{epoch+1}/{epochs}] Training"):
            images, gts = data 
            bboxes = gts["boxes"]
            labels = gts["labels"]

            optimizer.zero_grad()

            frcnn_labels, frcnn_bboxes, rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors= model(images)

            ith_rpn_loss = rpn_loss_function(rpn_predict_cls, rpn_predict_bbox_deltas, rpn_anchors)

            ith_frcnn_loss = frcnn_loss_function(frcnn_labels, frcnn_bboxes, labels, bboxes)

            rpn_loss += ith_rpn_loss
            frcnn_loss += ith_frcnn_loss

            total_loss = ith_frcnn_loss + ith_rpn_loss

            total_loss.backward()
            optimizer.step()

        rpn_loss /= configs.batch_number
        frcnn_loss /= configs.batch_number

        logger.write(epoch, RPN_Loss = rpn_loss, FRCNN_Loss = frcnn_loss)

        if best_loss > (rpn_loss + frcnn_loss):
            if not os.path.exists(configs.model_save_path):
                os.makedirs(configs.model_save_path) 
            torch.save(model.state_dict(), os.path.join(configs.model_save_path, "FRCNN_model.pth"))
            best_loss = rpn_loss + frcnn_loss

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