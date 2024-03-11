from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
import torch 
from PIL import Image
from glob import glob
import json
import xml.etree.ElementTree as ET

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
                    T.Normalize([])  # Add normalization parameters
                ])
            elif mode == "valid":
                self.transforms = T.Compose([
                    T.Resize((image_height, image_width)),
                    T.ToTensor(),
                    T.Normalize([])  # Add normalization parameters
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
        return img, {'boxes': bbox, 'labels': labels}
    
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

        image_id_to_path = {img['id']: os.path.join(self.root_dir, img['file_name']) for img in annotations['images']}
        image_paths = [image_id_to_path[img_id] for img_id in sorted(image_id_to_path)]
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

class YOLOv8(ObjectDetectionDataset):
    """
    Dataset class for YOLOv8 format annotations.

    Inherits from ObjectDetectionDataset and implements the __load_dataset__ and
    __parse_annotations__ methods specific to YOLOv8 format.
    """
    def __load_dataset__(self):
        """Loads the image file paths from the YOLOv8 annotation file."""
        with open(self.annotation_dir, 'r') as f:
            annotations = f.read().strip().split("\n")

        image_paths = [os.path.join(self.root_dir, line.split()[0]) for line in annotations]
        return image_paths

    def __parse_annotations__(self):
        """Parses the YOLOv8 annotations and maps them to the corresponding image IDs."""
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

class PascalVOCXML(ObjectDetectionDataset):
    """
    Dataset class for Pascal VOC XML format annotations.

    Inherits from ObjectDetectionDataset and implements the __load_dataset__,
    __parse_annotations__, and additional helper methods specific to Pascal VOC XML format.
    """
    def __init__(self, root_dir, image_height, image_width, annotation_dir, transforms=None, mode="train"):
        super().__init__(root_dir, image_height, image_width, annotation_dir, transforms, mode)
        self.id_to_category = self.__create_id_to_category_dictionary__()

    def __load_dataset__(self, recursive=False):
        """Loads the image file paths from the Pascal VOC dataset directory."""
        image_paths = glob(os.path.join(self.root_dir, "*.jpg"), recursive=recursive)
        return image_paths

    def __parse_annotations__(self):
        """Parses the Pascal VOC XML annotations and maps them to the corresponding image IDs."""
        annotation_files = glob(os.path.join(self.root_dir, "*.xml"))

        parsed_annotations = {image_id: [] for image_id in range(len(self.image_paths))}
        for index, xml_file in enumerate(annotation_files):
            parsed_annotations[index] = self.__parse_pascal_xml__(xml_file)

        return parsed_annotations

    def __create_id_to_category_dictionary__(self):
        """Creates a dictionary mapping category IDs to class names from the Pascal VOC XML annotations."""
        annotation_files = glob(os.path.join(self.root_dir, "*.xml"))
        id_to_cat = {}
        for xml_file in annotation_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                class_name = obj.find('name').text
                if class_name not in id_to_cat.values():
                    id_to_cat[len(id_to_cat)] = class_name

        return id_to_cat

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
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dl