import unittest
import torch 
from dataset import * 

class TestObjectDetectionDatasets(unittest.TestCase):

    def setUp(self):
        # Set up common parameters for all datasets
        self.image_height = 640
        self.image_width = 640

    def test_coco_dataset(self):
        # Specify the annotation directory for COCO dataset
        coco_annotation_dir = 'COCO/train/_annotations.coco.json'
        root_dir = "COCO/train"
        # Initialize COCO dataset
        coco_dataset = COCODataset(
            root_dir=root_dir,
            image_height=self.image_height,
            image_width=self.image_width,
            annotation_dir=coco_annotation_dir
        )
        # Fetch one item
        img, target = coco_dataset[0]
        # Check the format of the output
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(target, dict)
        self.assertIn('boxes', target)
        self.assertIn('labels', target)
        self.assertIsInstance(target['boxes'], torch.Tensor)
        self.assertIsInstance(target['labels'], torch.Tensor)

    def test_yolov8_dataset(self):
        # Initialize YOLOv8 dataset
        yolov8_dataset = YOLOv8(
            root_dir="YOLO/train",
            image_height=self.image_height,
            image_width=self.image_width,
            annotation_dir="YOLO/train/_annotations.txt"
        )
        # Fetch one item
        img, target = yolov8_dataset[0]
        # Check the format of the output
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(target, dict)
        self.assertIn('boxes', target)
        self.assertIn('labels', target)
        self.assertIsInstance(target['boxes'], torch.Tensor)
        self.assertIsInstance(target['labels'], torch.Tensor)

    def test_pascalvocxml_dataset(self):
        # Specify the annotation directory for Pascal VOC XML dataset
        pascalvocxml_annotation_dir = 'path_to_your_pascalvocxml_annotation_directory'
        # Initialize Pascal VOC XML dataset
        pascalvocxml_dataset = PascalVOCXML(
            root_dir="Pascal/train",
            image_height=self.image_height,
            image_width=self.image_width,
            annotation_dir=None
        )
        # Fetch one item
        img, target = pascalvocxml_dataset[0]
        # Check the format of the output
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(target, dict)
        self.assertIn('boxes', target)
        self.assertIn('labels', target)
        self.assertIsInstance(target['boxes'], torch.Tensor)
        self.assertIsInstance(target['labels'], torch.Tensor)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)