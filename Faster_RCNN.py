from rpn import *
import torch 
import torch.nn as nn 
from torchvision.models import resnet18, ResNet18_Weights
from roi import *

class Faster_RCNN(nn.Module): 

    def __init__(self, num_classes): 

        super(Faster_RCNN, self).__init__()

        self.backbone = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2])

        self.rpn = Regional_Proposal_Network(input_dimension=512, mid_dimension=256, conv_depth=4,
                                             score_threshold=0.7, nms_threshold=0.3, min_size=16,
                                             max_proposals=2000, size=(128, 256, 512),
                                             aspect_ratio=(0.5, 1.0, 2.0))
        
        self.roi = ROIAlign(output_size=(7, 7), scale=1.0, sampling_ratio = 2)
        
        self.detector_cls = nn.Sequential(*[nn.Linear(25088, num_classes), nn.Dropout(0.3)])

        self.detector_bbox = nn.Sequential(*[nn.Linear(25088, num_classes * 4), nn.Dropout(0.3)])

    def forward(self, x : torch.Tensor):

        feature_maps = self.backbone(x)

        roi = self.rpn(x, feature_maps)

        pooled_features = self.roi(feature_maps, roi).view(x.size(0), -1)

        print(pooled_features.shape)

        cls_label = self.detector_cls(pooled_features)

        bbox = self.detector_bbox(pooled_features)

        return cls_label, bbox