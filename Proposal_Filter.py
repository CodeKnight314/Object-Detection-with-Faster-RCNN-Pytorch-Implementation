import torch
import torch.nn as nn
from typing import Tuple

class ProposalFilter(nn.Module):
    def __init__(self, iou_threshold: float, min_size: int, score_threshold: float, max_proposals: int = 300):
        super(ProposalFilter, self).__init__()
        self.iou_threshold = iou_threshold
        self.min_size = min_size
        self.score_threshold = score_threshold
        self.max_proposals = max_proposals

    def forward(self, proposals: torch.Tensor, cls_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = proposals.shape[0]
        filtered_proposals = []
        filtered_scores = []
    
        for i in range(batch_size):
            prop = proposals[i]
            scores = cls_scores[i][:, 1]  # Assuming second column is objectness score
    
            # Score thresholding
            high_score_idxs = torch.where(scores > self.score_threshold)[0]
            prop = prop[high_score_idxs]
            scores = scores[high_score_idxs]
    
            # Apply NMS and select top-scoring proposals
            keep_idxs = self.nms(prop, scores)
            keep_idxs = keep_idxs[:self.max_proposals]
    
            # Check if we have less than max_proposals
            if len(keep_idxs) < self.max_proposals:
                pad_size = self.max_proposals - len(keep_idxs)
                if len(keep_idxs) == 0:
                    pad_idxs = torch.zeros(pad_size, dtype=torch.long, device=proposals.device)
                else:
                    pad_idxs = torch.full((pad_size,), keep_idxs[-1], dtype=torch.long, device=proposals.device)
                keep_idxs = torch.cat((keep_idxs, pad_idxs))
    
            # Ensure that keep_idxs has valid indices for prop
            keep_idxs = keep_idxs[keep_idxs < prop.shape[0]]
    
            # Check if prop is empty or keep_idxs is empty after NMS
            if prop.shape[0] == 0 or len(keep_idxs) == 0:
                filtered_proposals.append(torch.empty(0, 4, dtype=prop.dtype, device=prop.device))
                filtered_scores.append(torch.empty(0, dtype=scores.dtype, device=scores.device))
            else:
                filtered_proposals.append(prop[keep_idxs])
                filtered_scores.append(scores[keep_idxs])
    
        return torch.stack(filtered_proposals), torch.stack(filtered_scores)


    def nms(self, proposals: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        _, idxs = scores.sort(descending=True)
        proposals = proposals[idxs]

        keep = []
        while proposals.shape[0] > 0:
            keep.append(idxs[0].item())
            if proposals.shape[0] == 1:
                break
            ious = self.calculate_iou(proposals[0].unsqueeze(0), proposals[1:])
            idxs = idxs[1:][ious < self.iou_threshold]  # Keep idxs with IoU less than threshold
            proposals = proposals[1:][ious < self.iou_threshold]

        return torch.tensor(keep, dtype=torch.long, device=proposals.device)

    def calculate_iou(self, proposal: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        inter_area, union_area = self._calculate_areas(proposal, proposals)
        iou = inter_area / union_area
        return iou

    def _calculate_areas(self, proposal: torch.Tensor, proposals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, y1, x2, y2 = proposal[..., 0], proposal[..., 1], proposal[..., 2], proposal[..., 3]
        x1s, y1s, x2s, y2s = proposals[..., 0], proposals[..., 1], proposals[..., 2], proposals[..., 3]

        xi1 = torch.maximum(x1, x1s)
        yi1 = torch.maximum(y1, y1s)
        xi2 = torch.minimum(x2, x2s)
        yi2 = torch.minimum(y2, y2s)

        inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
        proposal_area = (x2 - x1) * (y2 - y1)
        proposals_area = (x2s - x1s) * (y2s - y1s)
        union_area = proposal_area + proposals_area - inter_area

        return inter_area, union_area
    
def main(): 
    batch_idx = 16
    num_of_proposals = 20 
    num_of_classes = 21 

    rpn_bbox = torch.rand((batch_idx, num_of_proposals, 4), dtype = torch.int64, device = "cuda" if torch.cuda.is_available() else "cpu")
    rpn_cls = torch.rand((batch_idx, num_of_proposals, num_of_classes), dtype = torch.float32, device = "cuda" if torch.cuda.is_available() else "cpu")

    proposal_filter = ProposalFilter(iou_threshold=0.7, 
                                     min_size=16, 
                                     score_threshold=0.5,
                                     max_proposals=100)
    
    filtered_bbox, fitlered_cls = proposal_filter(rpn_bbox, rpn_cls)

    print(filtered_bbox)
    print(fitlered_cls)

if __name__ == "__main__": 
    main()