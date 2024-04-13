from eval import *
from train import *
from dataset import * 
from loss import * 

def TAE(model : Faster_RCNN,
        train_dataLoader : DataLoader, 
        validation_dataLoader : DataLoader, 
        logger : LOGWRITER,
        optimizer : torch.optim, 
        scheduler : opt.lr_scheduler.StepLR,
        rpn_loss_function : RPNLoss,
        frcnn_loss_function : FasterRCNNLoss,
        epochs : int):
    
    for epoch in range(epochs): 
        
        train_batched_values = []
        for data in tqdm(train_dataLoader, desc = f"[Training: {epoch+1}/{epochs}]"): 
            values = train_step(model=model, 
                                data=data, 
                                optimizer=optimizer, 
                                rpn_loss_function=rpn_loss_function, 
                                frcnn_loss_function=frcnn_loss_function)
            train_batched_values.append(values)
        
        validation_batched_values = [] 
        for data in tqdm(validation_dataLoader, desc = f"[Validation: {epochs+1}/{epochs}]"): 
            values = eval_step(model=model,
                               data=data,
                               rpn_loss_function=rpn_loss_function,
                               frcnn_loss_function=frcnn_loss_function)
            validation_batched_values.append(values)

        averaged_train_values = torch.sum(torch.tensor(train_batched_values), dim = 1) / len(train_batched_values)
        averaged_valid_values = torch.sum(torch.tensor(validation_batched_values), dim = 1) / len(validation_batched_values)

        logger.write(epoch = epoch+1, 
                     Eval_RPN = averaged_valid_values[0], 
                     Eval_FRCNN = averaged_valid_values[1], 
                     Train_RPN = averaged_train_values[0], 
                     Train_FRCNN = averaged_train_values[1],
                     Precision = averaged_valid_values[5], 
                     Recall = averaged_valid_values[6], 
                     F1_score = averaged_valid_values[7],
                     Model_runtime = averaged_valid_values[4])
        
        if best_loss > (averaged_valid_values[0] + averaged_valid_values[1]):
            if not os.path.exists(configs.model_save_path):
                os.makedirs(configs.model_save_path) 
            torch.save(model.state_dict(), os.path.join(configs.model_save_path, f"FRCNN_model_{epoch+1}.pth"))
            best_loss = averaged_valid_values[0] + averaged_valid_values[1]
        
        scheduler.step()

def main(): 
    train_dataset = load_COCO_dataset(configs.root_dir,
                                      configs.image_height,
                                      configs.image_width, 
                                      configs.annotation_dir, 
                                      transforms=None,
                                      mode="train")
    
    train_dl = load_dataloaders(train_dataset[0], 
                                 configs.batch_number, 
                                 shuffle=True,
                                 drop_last=True)
    
    validation_dataset = load_COCO_dataset(configs.root_dir,
                                           configs.image_height,
                                           configs.image_width,
                                           configs.annotation_dir,
                                           transforms=None,
                                           mode="valid")
    
    val_dl = load_dataloaders(validation_dataset[0], 
                              configs.batch_number, 
                              shuffle=True,
                              drop_last=True)
    
    print("[INFO] Dataloaders loaded")
    print(f"[INFO] Total training samples {len(train_dataset[0])}.")
    print(f"[INFO] Total validation samples {len(validation_dataset[0])}.")

    model = get_model(cls_count=len(train_dataset[0].id_to_category), training=True)
    print(f"[INFO] Model loaded.")
    
    opt = get_optimizer(model, lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)
    scheduler = get_scheduler(optimizer=opt, step_size=configs.epochs//5, gamma=0.5)

    print("[INFO] Optimizer and scheduler loaded.")

    logger = LOGWRITER(configs.output_dir, configs.epochs)
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
        epochs = configs.epochs)

if __name__ == "__main__": 
    main()
