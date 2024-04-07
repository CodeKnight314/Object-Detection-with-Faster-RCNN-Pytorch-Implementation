from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
import torch 
from PIL import Image
from glob import glob
import json
import xml.etree.ElementTree as ET
from torch.nn.utils.rnn import pad_sequence
import configs

class ObjectDetectionDataset(Dataset, ABC):
    """
    Abstract base class for object detection datasets.

    Args:
        root_dir (str): Root directory of the dataset.
        image_height (int): Height to which the images will be resized.
        image_width (int): Width to which the images will be resized.
        annotation_dir (str): Directory containing the annotation files.
        transforms (torchvision.transforms.Compose, optional): Transformations to be applied to the images.
        mode (str): Mode of the dataset, can be 'train' or 'valid'.
    """
    def __init__(self, root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train"):
        super().__init__()
        self.root_dir = root_dir 
        self.annotation_dir = annotation_dir
        self.img_height = image_height 
        self.img_width = image_width
        self.transforms = transforms
        self.mode = mode

        if self.transforms is None:
            if mode == "train":
                self.transforms = T.Compose([
                    T.Resize((image_height, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization parameters
                ])
            elif mode == "valid":
                self.transforms = T.Compose([
                    T.Resize((image_height, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization parameters
                ])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.id_to_category = {}
        self.image_paths = self.__load_dataset__()
        self.annotations = self.__parse_annotations__()

    def __len__(self):
        """Returns the number of images in the dataset."""
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
        image = Image.open(self.image_paths[index]).convert("RGB")
        img_h, img_w = image.size

        x_scale_factor = self.img_width / img_w
        y_scale_factor = self.img_height / img_h

        img = self.transforms(image)
        ann = self.annotations[index]

        labels = []
        box = [] 

        for ann_item in ann: 
            for cat_id, bbox in ann_item.items():
                labels.append(cat_id)
                scaled_bbox = [
                    bbox[0] * x_scale_factor,  # xmin
                    bbox[1] * y_scale_factor,  # ymin
                    bbox[2] * x_scale_factor,  # xmax
                    bbox[3] * y_scale_factor   # ymax
                ]
                box.append(scaled_bbox)

        labels = torch.tensor(labels, dtype = torch.int64, device = self.device)
        box = torch.tensor(box, dtype = torch.float32, device = self.device)

        return img.to(self.device), {'boxes': box, 'labels': labels}
    
    def __load_dataset__(self):
        """Abstract method to load the dataset. Must be implemented by subclasses."""
        pass

    def __parse_annotations__(self):
        """Abstract method to parse annotations. Must be implemented by subclasses."""
        pass

class COCODataset(ObjectDetectionDataset):
    """
    Dataset class for COCO format annotations.

    Inherits from ObjectDetectionDataset and implements the __load_dataset__ and
    __parse_annotations__ methods specific to COCO format.
    """
    def __load_dataset__(self):
        """Loads the image file paths from the COCO annotation file."""
        with open(self.annotation_dir, 'r') as f:
            annotations = json.load(f)

        self.image_id_to_path = {img['id']: os.path.join(self.root_dir, img['file_name']) for img in annotations['images']}
        image_paths = [self.image_id_to_path[img_id] for img_id in sorted(self.image_id_to_path)]
        return image_paths
    
    def __parse_annotations__(self):
        """Parses the COCO annotations and maps them to the corresponding image IDs."""
        with open(self.annotation_dir, 'r') as f:
            annotations = json.load(f)

        # Mapping category IDs to class names
        for cls in annotations["categories"]:
            self.id_to_category[cls["id"]] = cls["name"]
        
        # Mapping annotations to image IDs
        ann = {img_id: [] for img_id in self.image_id_to_path}
        for annotation in annotations["annotations"]:
            image_id = annotation["image_id"]
            if image_id in ann:
                x, y, w, h = annotation["bbox"]
                bbox = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
                category = annotation["category_id"]
                ann[image_id].append({category: bbox})

        return ann

class YOLOv4(ObjectDetectionDataset):
    """
    Dataset class for YOLOv4 format annotations.

    Inherits from ObjectDetectionDataset and implements the __load_dataset__ and
    __parse_annotations__ methods specific to YOLOv4 format.
    """
    def __load_dataset__(self):
        """Loads the image file paths from the YOLOv4 annotation file."""
        with open(self.annotation_dir, 'r') as f:
            annotations = f.read().strip().split("\n")

        image_paths = [os.path.join(self.root_dir, line.split()[0]) for line in annotations]
        return image_paths

    def __parse_annotations__(self):
        """Parses the YOLOv4 annotations and maps them to the corresponding image IDs."""
        with open(self.annotation_dir, 'r') as f:
            annotations = f.read().split("\n")

        parsed_annotations = {img_id: [] for img_id in range(len(annotations))}
        for img_id, line in enumerate(annotations):
            parts = line.split()

            if len(parts) <= 1:
                continue

            for index, element in enumerate(parts):
                if index == 0:
                    continue
                coordinates = element.split(",")
                bbox = torch.tensor([int(x) for x in coordinates[:4]], dtype=torch.float32)
                cat_id = int(coordinates[4])
                parsed_annotations[img_id].append({cat_id: bbox})

        return parsed_annotations

class YOLOv5tov8(ObjectDetectionDataset):
    """
    Dataset class for converting YOLOv5 annotations to a format suitable for YOLOv8.

    Inherits from ObjectDetectionDataset and implements methods specific to handling
    YOLOv5 annotation format.
    """
    def __load_dataset__(self, recursive=False):
        """
        Loads the image file paths from the dataset directory.
        """
        image_paths = sorted(glob(os.path.join(self.root_dir, "images/*.jpg"), recursive=recursive))
        return image_paths
    
    def __parse_annotations__(self, recursive=False):
        """
        Parses the YOLOv5 annotations and converts them to a format suitable for YOLOv8.
        """
        annotations_dir = sorted(glob(os.path.join(self.root_dir, "labels/*.txt"), recursive=recursive))
        parsed_annotations = {img_id: [] for img_id in range(len(annotations_dir))}
        for img_id, (annotation_file, image_path) in enumerate(zip(annotations_dir, self.image_paths)):
            width, height = Image.open(image_path).convert("RGB").size
            with open(annotation_file, 'r') as f:
                annotation_lines = f.read().split("\n")
            
            for line in annotation_lines:
                if line.strip() == "":  # Skip empty lines
                    continue
                    
                parts = line.split(" ")
                x, y, w, h = [float(item) for item in parts[1:]]
                x_center_abs = x * width
                y_center_abs = y * height
                width_abs = w * width
                height_abs = h * height
                
                # Convert to [x_min, y_min, x_max, y_max] format
                bbox = torch.tensor([x_center_abs - width_abs/2, 
                                     y_center_abs - height_abs/2, 
                                     x_center_abs + width_abs/2, 
                                     y_center_abs + height_abs/2], dtype=torch.float32)
                
                class_id = int(parts[0])
                parsed_annotations[img_id].append({class_id: bbox})
        
        return parsed_annotations

class PascalVOCXML(ObjectDetectionDataset):
    """
    Dataset class for Pascal VOC XML format annotations.

    Inherits from ObjectDetectionDataset and implements the __load_dataset__,
    __parse_annotations__, and additional helper methods specific to Pascal VOC XML format.
    """

    def __load_dataset__(self, recursive=False):
        """Loads the image file paths from the Pascal VOC dataset directory."""
        image_paths = glob(os.path.join(self.root_dir, "*.jpg"), recursive=recursive)
        return image_paths

    def __parse_annotations__(self):
        """Parses the Pascal VOC XML annotations and maps them to the corresponding image IDs."""
        annotation_files = glob(os.path.join(self.root_dir, "*.xml"))
        self.id_to_category = {}

        for xml_file in annotation_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                class_name = obj.find('name').text
                if class_name not in self.id_to_category.values():
                    self.id_to_category[len(self.id_to_category)] = class_name
                    
        parsed_annotations = {image_id: [] for image_id in range(len(self.image_paths))}
        for index, xml_file in enumerate(annotation_files):
            parsed_annotations[index] = self.__parse_pascal_xml__(xml_file)

        return parsed_annotations

    def __parse_pascal_xml__(self, xml_file):
        """Parses a single Pascal VOC XML file and extracts object annotations."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        objects = []

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            class_id = {v: k for k, v in self.id_to_category.items()}[class_name]
            bbox = [
                int(obj.find('bndbox/xmin').text),
                int(obj.find('bndbox/ymin').text),
                int(obj.find('bndbox/xmax').text),
                int(obj.find('bndbox/ymax').text)
            ]
            objects.append({class_id: torch.tensor(bbox, dtype=torch.float32)})

        return objects

def load_COCO_dataset(root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train"): 
    return COCODataset(root_dir, image_height, image_width, annotation_dir, transforms=transforms, mode=mode)
    
def load_YOLOv4_dataset(root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train"):
    return YOLOv4(root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train")

def load_YOLOv5tov8_dataset(root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train"):
    return YOLOv5tov8(root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train")

def load_Pascal_dataset(root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train"): 
    return PascalVOCXML(root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train")

def load_dataloaders(dataset: ObjectDetectionDataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
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
    images, targets = list(zip(*batch))
    
    # Pad the bounding boxes and labels
    boxes = [item['boxes'] if item['boxes'].nelement() != 0 else torch.tensor([[-1, -1, -1, -1]], dtype=torch.float32) 
             for item in targets]
    labels = [item['labels'] if item['labels'].nelement() != 0 else torch.tensor([-1], dtype=torch.long) 
              for item in targets]

    for box in boxes: 
        print(box.shape)

    boxes = pad_sequence(boxes, batch_first=True, padding_value=-1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    # Create a new targets dictionary
    targets = [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, targets

def main(): 
    root_dir = "/workspace/train"
    annotation_dir = "/workspace/train/_annotations.coco.json"
    image_height = 600  # Example height
    image_width = 800  # Example width
    
    coco_dataset = COCODataset(root_dir, image_height, image_width, annotation_dir)
    
    # Debugging: Check the first few image paths and annotations
    for i in range(min(5, len(coco_dataset))):
        img, ann = coco_dataset[i]
        print(f'Image {i}: Path = {coco_dataset.image_paths[i]}, Annotations = {ann}\n')

    dl = DataLoader(coco_dataset, batch_size = configs.batch_number, shuffle = True, drop_last = True, collate_fn = collate_fn)

    batch = next(iter(dl))

if __name__ == "__main__": 
    main()