from rpn import *
import torch 
import torch.nn as nn 
from torchvision.models import resnet18, ResNet18_Weights
from roi import *
import configs 
import time

class Faster_RCNN(nn.Module): 
    def __init__(self, num_classes : int, train_mode : bool = True): 

        super(Faster_RCNN, self).__init__()

        self.num_classes = num_classes

        self.backbone = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2])

        self.rpn = Regional_Proposal_Network(input_dimension=512, mid_dimension=256, conv_depth=4,
                                             score_threshold=0.5, iou_threshold=0.3, min_size=16,
                                             max_proposals=256, size=(128, 256, 512),
                                             aspect_ratio=(0.5, 1.0, 2.0))
        
        self.roi = ROIAlign(output_size=(7, 7), spatial_scale=1.0)

        self.input_feature_dim = 512 * self.roi.output_size[0] * self.roi.output_size[1]
        
        self.detector_cls = nn.Sequential(*[nn.Linear(self.input_feature_dim, num_classes), nn.Dropout(0.3), nn.Sigmoid()])

        self.detector_bbox = nn.Sequential(*[nn.Linear(self.input_feature_dim, num_classes * 4), nn.Dropout(0.3)])

        self.train_mode = train_mode

        self.time_records = {}

    def forward(self, x : torch.Tensor):
        """
        Forward pass of the Faster R-CNN model.

        Args:
            x (torch.Tensor): The input tensor representing a batch of images with shape (batch_size, channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors.
                - The first tensor is the classification labels for each ROI with shape (batch_size, num_proposals, num_classes),
                where num_classes includes the background class.
                - The second tensor is the bounding box regressions for each ROI with shape (batch_size, num_proposals, num_classes * 4),
                where each set of 4 values represents the (x_min, y_min, x_max, y_max) adjustments for the corresponding class.
        """
        self.time_records = {}
        feature_maps = self.backbone(x)

        rpn_start = time.time() 
        if self.train_mode:
            roi, predict_cls, predict_bbox_deltas, anchors = self.rpn(x, feature_maps)

        else:
            roi = self.rpn(x, feature_maps)
        rpn_end = time.time()

        self.time_records["RPN"] = rpn_end - rpn_start

        roi_start = time.time()
        pooled_features = self.roi(feature_maps, roi) # (N*P, 512, 7, 7)
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        roi_end = time.time() 
        self.time_records["ROI"] = roi_end - roi_start

        detectors_start = time.time()
        cls_label = self.detector_cls(pooled_features)

        bbox = self.detector_bbox(pooled_features)
        detectors_end = time.time()
        self.time_records["Detectors"] = detectors_end - detectors_start

        self.time_records["Total"] = detectors_end - rpn_start

        if self.train_mode: 
            return cls_label.reshape(x.size(0), -1, self.num_classes), bbox.reshape(x.size(0), -1, self.num_classes, 4), predict_cls, predict_bbox_deltas, anchors
        else:
            return cls_label, bbox

def get_model(cls_count : int, training : bool = True): 
    """
    Helper function to get model
    """
    return Faster_RCNN(cls_count, training).to("cuda" if torch.cuda.is_available() else "cpu")

def main(): 
    batch_idx = 16 
    image_channel = 3 
    image_height = 640 
    image_width = 640 

    num_of_classes = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = torch.rand((batch_idx, image_channel, image_height, image_width), dtype = torch.float32, device = device)

    model = Faster_RCNN(num_of_classes).to(device)

    cls_labels, bboxes = model(image)

    print(cls_labels.shape)

    print(bboxes.shape)

if __name__ == "__main__": 
    main()