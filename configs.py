model_path = None

batch_number = 32

lr = 1e-4 
betas = ((0.9, 0.999))
weight_decay = 1e-3

epochs = 25 

model_save_path = "/workspace/Model_save_dir/"

num_of_classes = None

root_dir = "/workspace/train"

image_height = 640 
image_width = 640 

annotation_dir = "/workspace/train/_annotations.coco.json"

transforms = None

mode = "train"

train_mode = True

output_dir = "/workspace/Log_output_dir/"