import torch
import torch.nn as nn
from typing import Tuple

class AnchorGenerator(nn.Module):
    """
    Generates anchor boxes for a given set of sizes and aspect ratios.

    Attributes:
        sizes (Tuple[int]): A tuple of integers representing the sizes of the anchor boxes.
        aspect_ratios (Tuple[float]): A tuple of floats representing the aspect ratios of the anchor boxes.
    """
    def __init__(self, sizes: Tuple[int], aspect_ratios: Tuple[float]) -> None:
        super(AnchorGenerator, self).__init__()
        assert len(sizes) == len(aspect_ratios), f"[Error] AnchorGenerator tuple sizes do not match. sizes : {len(sizes)}, aspect ratios: {len(aspect_ratios)}"
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = self.generate_cell_anchors(self.sizes, self.aspect_ratios)

    def generate_cell_anchors(self, sizes: Tuple[int], aspect_ratios: Tuple[float], dtype: torch.dtype = torch.float32, device="cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
        """
        Generates anchor boxes for each cell in the feature map.

        Args:
            sizes (Tuple[int]): A tuple of integers representing the sizes of the anchor boxes.
            aspect_ratios (Tuple[float]): A tuple of floats representing the aspect ratios of the anchor boxes.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float32.
            device (str, optional): Device where the tensors will be stored. Defaults to "cuda" if available, else "cpu".

        Returns:
            torch.Tensor: A tensor containing the generated anchor boxes with shape (num_anchors, 4).
        """
        size_tensors = torch.tensor(sizes, dtype=dtype, device=device)
        ar_tensors = torch.tensor(aspect_ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ar_tensors)
        w_ratios = 1 / h_ratios

        hs = (h_ratios[:, None] * size_tensors[None, :]).view(-1)
        ws = (w_ratios[:, None] * size_tensors[None, :]).view(-1)

        cell_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return cell_anchors.round()

    def number_of_anchors_per_location(self) -> int:
        """
        Returns the number of anchors per location.

        Returns:
            int: The number of anchors per location.
        """
        return len(self.sizes) * len(self.aspect_ratios)

    def forward(self, images: torch.Tensor, feature_maps: torch.Tensor) -> torch.Tensor:
        """
        Generates anchors for the entire image.

        Args:
            images (torch.Tensor): The input images with shape (batch_size, channels, height, width).
            feature_maps (torch.Tensor): The feature maps from the backbone network with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor containing the generated anchors for each feature map with shape (batch_size, num_anchors, 4).
        """
        image_dim = [(images.size(2), images.size(3)) for _ in range(images.size(0))]
        feature_dim = [(feature_maps.size(2), feature_maps.size(3)) for _ in range(feature_maps.size(0))]
        strides = [(img_dim[0] / f_dim[0], img_dim[1] / f_dim[1]) for img_dim, f_dim in zip(image_dim, feature_dim)]
        dtype, device = feature_maps.dtype, feature_maps.device

        anchors = torch.stack([self.generate_image_anchors(f_dim, stride, feature_maps.dtype, feature_maps.device) for f_dim, stride in zip(feature_dim, strides)], dim=0)
        return anchors

    def generate_image_anchors(self, feature_map_size: Tuple[int, int], stride: Tuple[float, float], dtype: torch.dtype, device: str) -> torch.Tensor:
        """
        Generates anchors for a single feature map.

        Args:
            feature_map_size (Tuple[int, int]): The size of the feature map (height, width).
            stride (Tuple[float, float]): The stride of the feature map relative to the original image size.
            dtype (torch.dtype): Data type of the tensors.
            device (str): Device where the tensors will be stored.

        Returns:
            torch.Tensor: A tensor containing the generated anchors for the feature map with shape (num_anchors, 4).
        """
        grid_height, grid_width = feature_map_size
        shifts_x = torch.arange(0, grid_width, dtype=dtype, device=device) * stride[0]
        shifts_y = torch.arange(0, grid_height, dtype=dtype, device=device) * stride[1]
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack((-shift_x, -shift_y, shift_x, shift_y), dim=-1).view(-1, 4)

        stride_tensor = torch.tensor(stride, dtype=dtype, device=device).view(1, 2).repeat(1, 2)
        scaled_anchors = self.cell_anchors * stride_tensor

        all_anchors = scaled_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        return all_anchors.view(-1, 4)