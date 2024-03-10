from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
import torch 
from PIL import Image
from glob import glob
import json

class ObjectDetectionDataset(Dataset, ABC): 

    def __init__(self, root_dir, image_height, image_width, annotation_dir, transforms = None, mode = "train"):
        super().__init__()
        self.root_dir = root_dir 
        self.annotation_dir = annotation_dir
        self.img_height = image_height 
        self.img_width = image_width

        self.transforms = transforms

        if mode == "train" and transforms is None: 
            self.transform = T.Compose([
                T.Resize((image_height, image_width)), 
                T.ToTensor(), 
                T.Normalize([])
            ])

        elif mode == "valid" and transforms is None: 
            self.transforms = T.Compose([
                T.ToTensor(), 
                T.Normalize([])
            ])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.id_to_category = {}
        
        self.image_paths = self.__load_dataset__()
        self.annotations = self.__parse_annotations__()

    def __len__(self): 
        return len(self.image_paths)
    
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
        img = self.transforms(Image.open(self.image_paths[index]).convert("RGB"))
        ann = self.annotations[index]

        labels = []
        bbox = [] 

        for ann_item in ann: 
            for cat_id, bbox in ann_item.items():
                labels.append(cat_id)            
                bbox.append(bbox)

        bbox = torch.stack(bbox)
        labels = torch.tensor(labels)
        return img, bbox, labels
    
    def __load_dataset__(self): 
        pass

    def __parse_annotations__(self): 
        pass

class COCODataset(ObjectDetectionDataset): 
    
    def __load_dataset__(self):
        with open(self.annotation_dir, 'r') as f:
            annotations = json.load(f)

        image_id_to_path = {img['id']: os.path.join(self.root_dir, img['file_name']) for img in annotations['images']}
        image_paths = [self.image_id_to_path[img_id] for img_id in sorted(image_id_to_path)]
        return image_paths
    
    def __parse_annotations__(self):
        path = self.annotation_dir 
        with open(path, 'r') as f: 
            annotations = json.load(f)
        
        # Mapping the category ids to classes
        for cls in annotations["category"]: 
            self.id_to_category[cls["id"]] = cls["name"]
        
        # Each ls entry is formatted as {Image_id : [{category_id : torch.tensor(x_min, y_min, x_max, y_max)}]}
        ann = {img_id : [] for img_id in self.image_id_to_path}
        for annotation in annotations["annotations"]: 
            image_id = annotation["image_id"]
            if image_id in ann: 
                x, y, w, h = annotation["bbox"]
                x_min, y_min = x, y 
                x_max, y_max = x+w, y+h
                category = annotation["category_id"]
                ann[image_id].append({category : torch.tensor([x_min, y_min, x_max, y_max], dtype = torch.float32)})

        return ann

class YOLOv8(ObjectDetectionDataset): 

    def __load_dataset__(self):
        with open(self.annotation_dir, 'r') as f: 
            annotations = f.read().strip().split("\n")
        
        image_paths = [os.path.join(self.root_dir, lines.split()[0]) for lines in annotations]
        return image_paths

    def __parse_annotations__(self):
        with open(self.annotation_dir, 'r') as f: 
            annotations = f.read().split("\n")

        parsed_annotations = {img_id : [] for img_id in range(len(annotations))}
        for img_id, line in enumerate(annotations): 
            parts = line.split() 

            if len(parts) <= 1: 
                parsed_annotations[img_id] = []
                continue

            for index, element in enumerate(parts):
                if index == 0: 
                    continue
                coordinates = element.split(",")
                bbox = torch.tensor([int(x) for x in coordinates[:4]], dtype = torch.float32)
                cat_id = coordinates[4]
                parsed_annotations[img_id].append({cat_id : bbox})
        
        return parsed_annotations
    
class PascalVOCXML(ObjectDetectionDataset): 

    def __load_dataset__(self, recursive = False):
        img_paths = sorted(glob(self.root_dir + "/*.jpg", recursive=recursive))
        return img_paths 

    def __parse_annotations__(self):
        pass


def load_dataloaders(dataset : ObjectDetectionDataset, batch_size : int, shuffle : bool, drop_last : bool) -> DataLoader:
    dl = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)
    return dl
