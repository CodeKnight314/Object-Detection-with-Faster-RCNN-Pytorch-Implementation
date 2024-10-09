from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
import torch 
from PIL import Image
import json
from torch.nn.utils.rnn import pad_sequence
import argparse

class HumanDetectionDataset(Dataset):
    """
    Abstract base class for object detection datasets.

    Args:
        root_dir (str): Root directory of the dataset.
        image_height (int): Height to which the images will be resized.
        image_width (int): Width to which the images will be resized.
        transforms (torchvision.transforms.Compose, optional): Transformations to be applied to the images.
        mode (str): Mode of the dataset, can be 'train' or 'valid'.
    """
    def __init__(self, root_dir, image_height, image_width, transforms=None, mode="train"):
        super().__init__()
        self.root_dir = root_dir 
        self.data_dir = os.path.join(self.root_dir, mode)
        self.img_height = image_height 
        self.img_width = image_width
        self.transforms = transforms
        self.mode = mode

        if self.transforms is None:
            self.transforms = T.Compose([
                T.Resize((image_height, image_width)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization parameters
            ])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.id_to_category = {}
        self.img_ids = self.__load_dataset__()
        self.annotations = self.__parse_annotations__()

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        Fetch the image and corresponding annotations for the given index.

        Args:
            index (int): Index of the dataset element to fetch.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - First element is the image tensor of shape (C, H, W).
                - Second element is a dict containing:
                  - 'boxes': a tensor of bounding boxes of shape (N, 4),
                  - 'labels': a tensor of the corresponding labels of shape (N,),
                  where N is the number of objects in the image.
        """
        id = self.img_ids[index]
        image = Image.open(self.image_id_to_path[id]).convert("RGB")
        img_h, img_w = image.size

        x_scale_factor = self.img_width / img_w
        y_scale_factor = self.img_height / img_h

        img = self.transforms(image)
        ann = self.annotations[id]

        box = [] 

        for ann_item in ann: 
            for _, bbox in ann_item.items():
                scaled_bbox = [
                    bbox[0] * x_scale_factor,  # xmin
                    bbox[1] * y_scale_factor,  # ymin
                    bbox[2] * x_scale_factor,  # xmax
                    bbox[3] * y_scale_factor   # ymax
                ]
                box.append(scaled_bbox)

        box = torch.tensor(box, dtype = torch.float32, device = self.device)
        labels = torch.tensor([1] * len(box), dtype=torch.int64)
        
        return img.to(self.device), {'boxes': box, "labels" : labels}
    
    def __load_dataset__(self):
        """Loads the image file paths from the COCO annotation file."""
        with open(os.path.join(self.data_dir, f"person_keypoints_{self.mode}2017.json"), 'r') as f:
            annotations = json.load(f)

        self.image_id_to_path = {img['id']: os.path.join(self.data_dir, img['file_name']) for img in annotations['images']}
        img_ids = [img_id for img_id in sorted(self.image_id_to_path)]
        return img_ids
    
    def __parse_annotations__(self):
        """Parses the COCO annotations and maps them to the corresponding image IDs."""
        with open(os.path.join(self.data_dir,f"person_keypoints_{self.mode}2017.json"), 'r') as f:
            annotations = json.load(f)

        # Mapping category IDs to class names
        for cls in annotations["categories"]:
            self.id_to_category[cls["id"]] = cls["name"]
        
        # Mapping annotations to image IDs, only keeping those corresponding to humans
        human_category_id = [cls["id"] for cls in annotations["categories"] if cls["name"].lower() == "person"]
        if not human_category_id:
            raise ValueError("No category with the name 'person' found in the annotations.")
        human_category_id = human_category_id[0]

        ann = {img_id: [] for img_id in self.image_id_to_path}
        for annotation in annotations["annotations"]:
            if annotation["category_id"] == human_category_id:
                image_id = annotation["image_id"]
                if image_id in ann:
                    x, y, w, h = annotation["bbox"]
                    bbox = [x, y, x + w, y + h]
                    category = annotation["category_id"]
                    ann[image_id].append({category: bbox})

        return ann

def load_dataloaders(dataset: HumanDetectionDataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    """
    Utility function to create a DataLoader from an ObjectDetectionDataset.

    Args:
        dataset (ObjectDetectionDataset): The dataset to load into the DataLoader.
        batch_size (int): The number of samples in each batch.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
        DataLoader: A DataLoader for the given dataset.
    """
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn = collate_fn)
    return dl

def collate_fn(batch):
    """
    Custom collate function for batching images and their corresponding target dictionaries which include bounding boxes
    and labels. This function handles cases where bounding boxes or labels might be empty by assigning placeholder values.

    Args:
        batch (list of tuples): A list of tuples, where each tuple contains an image tensor and a target dictionary. The
                                target dictionary for each image includes 'boxes' and 'labels'.

    Returns:
        Tuple[torch.Tensor, list of dict]: A tuple containing a batched tensor of images and a list of target dictionaries.
                                           The images tensor has a shape of (N, C, H, W).
    """
    images, targets = list(zip(*batch))
    
    # Pad the bounding boxes and labels
    boxes = [item['boxes'] if item['boxes'].nelement() != 0 else torch.tensor([[-1, -1, -1, -1]], dtype=torch.float32) 
             for item in targets]
    labels = [item['labels'] if item['labels'].nelement() != 0 else torch.tensor([-1], dtype=torch.long) 
              for item in targets]

    boxes = pad_sequence(boxes, batch_first=True, padding_value=-1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    # Create a new targets dictionary
    targets = [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, targets

def main(args): 
    coco_dataset = HumanDetectionDataset(args.root_dir, args.img_height, args.img_width, args.mode)
    
    img, target = coco_dataset.__getitem__(index=0)
    
    print(f"Image shape: {img.shape}")
    print(f"bbox shape: {target["boxes"].shape}")
    print(f"labels shape: {target["labels"].shape}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--mode", type=str, default="train")
    
    args = parser.parse_args()
    
    main(args)