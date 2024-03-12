import torch
from typing import Tuple, List, Union

def calculate_iou(proposals: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """
    Calculate IOU for a set of proposals with a set of references.

    Args:
        proposals (torch.Tensor): The proposal anchors with shape (number of proposals, 4).
        references (torch.Tensor): The reference anchors with shape (number of references, 4).

    Returns:
        torch.Tensor: Tensor containing the IOU values with shape (number of proposals, number of references).
    """
    # Calculate the intersection
    max_xy = torch.min(proposals[:, None, 2:], references[:, 2:])
    min_xy = torch.max(proposals[:, None, :2], references[:, :2])
    intersection = torch.clamp(max_xy - min_xy, min=0)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    # Calculate the union
    proposals_area = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    references_area = (references[:, 2] - references[:, 0]) * (references[:, 3] - references[:, 1])
    union_area = proposals_area[:, None] + references_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def bbox_encode(anchors: torch.Tensor, anchor_offsets: torch.Tensor) -> torch.Tensor:
    """
    Decodes the bounding box deltas and applies them to the anchor boxes.

    Args:
        anchors (torch.Tensor): The anchor boxes tensor with shape (batch_size, num_anchors, 4).
            Each anchor box is represented as (x_min, y_min, x_max, y_max).
        anchor_offsets (torch.Tensor): The predicted bounding box deltas with shape (batch_size, num_anchors, 4).
            Each delta is represented as (dx, dy, dw, dh).

    Returns:
        torch.Tensor: The adjusted anchor boxes tensor with shape (batch_size, num_anchors, 4).
            Each adjusted box is represented as (x_min, y_min, x_max, y_max).
    """
    # Calculate the widths and heights of the anchors
    anchor_widths = anchors[:, :, 2] - anchors[:, :, 0]
    anchor_heights = anchors[:, :, 3] - anchors[:, :, 1]

    # Calculate the center coordinates of the anchors
    anchor_center_x = anchors[:, :, 0] + 0.5 * anchor_widths
    anchor_center_y = anchors[:, :, 1] + 0.5 * anchor_heights

    # Apply the offsets to the anchor center coordinates
    new_center_x = anchor_offsets[:, :, 0] * anchor_widths + anchor_center_x
    new_center_y = anchor_offsets[:, :, 1] * anchor_heights + anchor_center_y

    # Apply the scaling factors to the widths and heights
    new_widths = torch.exp(anchor_offsets[:, :, 2]) * anchor_widths
    new_heights = torch.exp(anchor_offsets[:, :, 3]) * anchor_heights

    # Calculate the top-left and bottom-right coordinates of the adjusted anchors
    top_left_x = new_center_x - 0.5 * new_widths
    top_left_y = new_center_y - 0.5 * new_heights
    bottom_right_x = new_center_x + 0.5 * new_widths
    bottom_right_y = new_center_y + 0.5 * new_heights

    # Combine the coordinates into a single tensor
    adjusted_anchors = torch.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim=2)

    return adjusted_anchors

def bbox_decode(anchor_batch_A: torch.Tensor, anchor_batch_B: torch.Tensor) -> torch.Tensor:
    """
    Encodes the relative deltas between two batches of anchor boxes.

    Args:
        anchor_batch_A (torch.Tensor): The first batch of anchor boxes with shape (batch_size, num_anchors, 4).
            Each anchor box is represented as (x_min, y_min, x_max, y_max).
        anchor_batch_B (torch.Tensor): The second batch of anchor boxes with shape (batch_size, num_anchors, 4).
            Each anchor box is represented as (x_min, y_min, x_max, y_max).

    Returns:
        torch.Tensor: The relative deltas tensor with shape (batch_size, num_anchors, 4).
            Each delta is represented as (dx, dy, dw, dh).
    """
    # Calculate the widths and heights of the anchors in both batches
    widths_A = anchor_batch_A[:, :, 2] - anchor_batch_A[:, :, 0]
    heights_A = anchor_batch_A[:, :, 3] - anchor_batch_A[:, :, 1]
    widths_B = anchor_batch_B[:, :, 2] - anchor_batch_B[:, :, 0]
    heights_B = anchor_batch_B[:, :, 3] - anchor_batch_B[:, :, 1]

    # Calculate the center coordinates of the anchors in both batches
    center_x_A = anchor_batch_A[:, :, 0] + 0.5 * widths_A
    center_y_A = anchor_batch_A[:, :, 1] + 0.5 * heights_A
    center_x_B = anchor_batch_B[:, :, 0] + 0.5 * widths_B
    center_y_B = anchor_batch_B[:, :, 1] + 0.5 * heights_B

    # Compute the relative deltas
    dx = (center_x_B - center_x_A) / widths_A
    dy = (center_y_B - center_y_A) / heights_A
    dw = torch.log(widths_B / widths_A)
    dh = torch.log(heights_B / heights_A)

    # Combine the deltas into a single tensor
    deltas = torch.stack([dx, dy, dw, dh], dim=2)

    return deltas

def CWH_to_TLBR(coordinates: Tuple[float, float, float, float], 
                reversed: bool, 
                normalize: bool, 
                image_dimensions: Union[Tuple[int, int], None] = None) -> List[float]:
    """
    Convert bounding box coordinates from center-width-height (CWH) format to top-left-bottom-right (TLBR) format.

    Args:
        coordinates (Tuple[float, float, float, float]): Bounding box coordinates in CWH format.
        reversed (bool): If True, convert from TLBR to CWH format.
        normalize (bool): If True, normalize the coordinates based on image dimensions.
        image_dimensions (Union[Tuple[int, int], None]): Dimensions of the image (width, height).

    Returns:
        List[float]: Bounding box coordinates in TLBR format.
    """
    if normalize and image_dimensions is None:
        raise ValueError("[ERROR] image_dimensions must be provided when normalize is True")

    if image_dimensions:
        img_width, img_height = image_dimensions

    if reversed:
        x1, y1, x2, y2 = coordinates
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        if normalize:
            return [center_x / img_width, center_y / img_height, width / img_width, height / img_height]
        else:
            return [center_x, center_y, width, height]
    else:
        center_x, center_y, width, height = coordinates
        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2
        bottom_right_x = center_x + width / 2
        bottom_right_y = center_y + height / 2

        if normalize:
            return [top_left_x / img_width, top_left_y / img_height, bottom_right_x / img_width, bottom_right_y / img_height]
        else:
            return [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

def CWH_to_TLWH(coordinates : Tuple[float, float, float, float],
                reversed: bool, 
                normalize: bool, 
                image_dimensions: Union[Tuple[int, int], None] = None) -> List[float]: 
    """
    Convert bounding box coordinates from center-width-height (CWH) format to top-left-width-height (TLWH) format.

    Args:
        coordinates (Tuple[float, float, float, float]): Bounding box coordinates in CWH format.
        reversed (bool): If True, convert from TLWH to CWH format.
        normalize (bool): If True, normalize the coordinates based on image dimensions.
        image_dimensions (Union[Tuple[int, int], None]): Dimensions of the image (width, height).

    Returns:
        List[float]: Bounding box coordinates in TLWH format.
    """
    if normalize and image_dimensions is None: 
        raise ValueError("[ERROR] image_dimensions must be provided when normalize is True")
    
    if image_dimensions: 
        img_width, img_height = image_dimensions 
    
    if reversed: 
        x1, y1, width, height = coordinates 
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        if normalize:
            return [center_x / img_width, center_y / img_height, width / img_width, height / img_height]
        else:
            return [center_x, center_y, width, height]
    else: 
        center_x, center_y, width, height = coordinates
        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2

        if normalize:
            return [top_left_x / img_width, top_left_y / img_height, width / img_width, height / img_height]
        else:
            return [top_left_x, top_left_y, width, height]
        
def TLWH_to_TLBR(coordinates : Tuple[float, float, float, float],
                reversed: bool, 
                normalize: bool, 
                image_dimensions: Union[Tuple[int, int], None] = None) -> List[float]: 
    """
    Convert bounding box coordinates from top-left-width-height (TLWH) format to top-left-bottom-right (TLBR) format.

    Args:
        coordinates (Tuple[float, float, float, float]): Bounding box coordinates in TLWH format.
        reversed (bool): If True, convert from TLBR to TLWH format.
        normalize (bool): If True, normalize the coordinates based on image dimensions.
        image_dimensions (Union[Tuple[int, int], None]): Dimensions of the image (width, height).

    Returns:
        List[float]: Bounding box coordinates in TLBR format.
    """
    if normalize and image_dimensions is None: 
        raise ValueError("[ERROR] image_dimensions must be provided when normalize is True")
    
    if image_dimensions: 
        img_width, img_height = image_dimensions

    if reversed: 
        x1, y1, x2, y2 = coordinates
        width = x2 - x1
        height = y2 - y1
        
        if normalize: 
            return [x1 / img_width, y1 / img_height, width / img_width, height / img_height]
        else: 
            return [x1, y1, width, height]
        
    else:
        x1, y1, width, height = coordinates
        x2 = x1 + width
        y2 = y1 + height

        if normalize:
            return [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height]
        else:
            return [x1, y1, x2, y2]