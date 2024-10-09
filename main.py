import torch
import os
from tqdm import tqdm
from eval import eval_step
from train import train_step
from dataset import get_dataset
from loss import get_loss_functions
from Faster_RCNN import get_model
from utils.log_writer import LOGWRITER

def TAE(model: torch.nn.Module,
        train_dataLoader: torch.utils.data.DataLoader,
        validation_dataLoader: torch.utils.data.DataLoader,
        logger: LOGWRITER,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        rpn_loss_function,
        frcnn_loss_function,
        epochs: int,
        output_path: str):
    """
    Trains and evaluates the Faster R-CNN model for a given set of train and validation dataset

    Args: 
        model (torch.nn.Module): Faster R-CNN model for evaluation. Model path is loaded if valid. 
        train_dataLoader (DataLoader): Train dataset under the subclass of ObjectDetectionDataset. 
        validation_dataLoader (DataLoader): Validation dataset under subclass of ObjectDetectionDataset.
        logger (LOGWRITER): Log writer that takes kwargs and writes them to txt file.
        optimizer (torch.optim.Optimizer): Optimizer for model to update and propagate loss.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler to update the learning rate of the optimizer
        rpn_loss_function: RPN Loss function for calculating loss 
        frcnn_loss_function: FRCNN Loss function for calculating loss 
        epochs (int): Total number of epochs
    """

    best_loss = float('inf')
    
    for epoch in range(epochs): 
        model.train()
        train_batched_values = []
        for data in tqdm(train_dataLoader, desc=f"[Training: {epoch+1}/{epochs}]"): 
            values = train_step(model=model, 
                                data=data, 
                                optimizer=optimizer, 
                                rpn_loss_function=rpn_loss_function, 
                                frcnn_loss_function=frcnn_loss_function)
            train_batched_values.append(values)
        
        model.eval()
        validation_batched_values = [] 
        with torch.no_grad():
            for data in tqdm(validation_dataLoader, desc=f"[Validation: {epoch+1}/{epochs}]"): 
                values = eval_step(model=model,
                                   data=data,
                                   rpn_loss_function=rpn_loss_function,
                                   frcnn_loss_function=frcnn_loss_function)
                validation_batched_values.append(values)

        averaged_train_values = torch.sum(torch.tensor(train_batched_values), dim=0) / len(train_batched_values)
        averaged_valid_values = torch.sum(torch.tensor(validation_batched_values), dim=0) / len(validation_batched_values)

        logger.write(epoch=epoch+1, 
                     Eval_RPN=averaged_valid_values[0].item(), 
                     Eval_FRCNN=averaged_valid_values[1].item(), 
                     Train_RPN=averaged_train_values[0].item(), 
                     Train_FRCNN=averaged_train_values[1].item(),
                     Precision=averaged_valid_values[5].item(), 
                     Recall=averaged_valid_values[6].item(), 
                     F1_score=averaged_valid_values[7].item(),
                     Model_runtime=averaged_valid_values[4].item())
        
        current_loss = averaged_valid_values[0] + averaged_valid_values[1]
        if best_loss > (current_loss[0] + current_loss[1]):
            if not os.path.exists(output_path):
                os.makedirs(output_path) 
            torch.save(model.state_dict(), os.path.join(output_path, f"FRCNN_model_{epoch+1}.pth"))
            best_loss = current_loss[0] + current_loss[1]
        
        scheduler.step()

def main(args): 
    train_dl = get_dataset(args.root_dir, args.img_h, args.img_w, "train", args.batch_size)
    val_dl = get_dataset(args.root_dir, args.img_h, args.img_w, "val", args.batch_size)
    
    print("[INFO] Dataloaders loaded")
    print(f"[INFO] Total training samples: {len(train_dl)}.")
    print(f"[INFO] Total validation samples: {len(val_dl)}.")

    model = get_model(cls_count=2, training=True)  # cls_count=2 since it's either human or not human
    print(f"[INFO] Faster-RCNN Model loaded.")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, eta_min=args.lr / 100, T_max=args.epochs * 0.75)
    print("[INFO] Optimizer and scheduler loaded.")

    logger = LOGWRITER(args.output_dir, args.epochs)
    rpn_loss, frcnn_loss = get_loss_functions((0.7, 0.3))
    print("[INFO] Utility functions loaded.")

    TAE(model=model,
        train_dataLoader=train_dl, 
        validation_dataLoader=val_dl,
        logger=logger,
        optimizer=opt,
        scheduler=scheduler,
        rpn_loss_function=rpn_loss, 
        frcnn_loss_function=frcnn_loss, 
        epochs=args.epochs)

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--img_h', type=int, default=800, help='Height of the input image')
    parser.add_argument('--img_w', type=int, default=800, help='Width of the input image')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=20, help='Total number of epochs to train')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and model checkpoints')
    args = parser.parse_args()
    
    main(args)