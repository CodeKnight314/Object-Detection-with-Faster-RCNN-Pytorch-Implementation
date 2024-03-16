from rpn import *
import torch 
import torch.nn as nn 
from torchvision.models import resnet18, ResNet18_Weights
from roi import *

class Faster_RCNN(nn.Module): 

    def __init__(self, num_classes): 

        super(Faster_RCNN, self).__init__()

        self.backbone = nn.Sequential(*list(resnet18(weights = ResNet18_Weights).children())[:-2])

        self.rpn = Regional_Proposal_Network(input_dimension=512, mid_dimension=256, conv_depth=4,
                                             score_threshold=0.7, nms_threshold=0.3, min_size=16,
                                             max_proposals=2000, size=(128, 256, 512),
                                             aspect_ratio=(0.5, 1.0, 2.0))
        
        self.roi = ROIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        
        self.detector_cls = nn.Sequential(*[nn.Linear(512, num_classes), nn.Dropout(0.3)])

        self.detector_bbox = nn.Sequential(*[nn.Linear(512, num_classes * 4), nn.Dropout(0.3)])

    def forward(self, x : torch.Tensor):

        feature_maps = self.backbone(x)

        roi = self.rpn(x, feature_maps)

        filtered_roi = self.roi(roi)

        cls_label = self.detector_cls(filtered_roi)

        bbox = self.detector_bbox(filtered_roi)

        return cls_label, bbox


        