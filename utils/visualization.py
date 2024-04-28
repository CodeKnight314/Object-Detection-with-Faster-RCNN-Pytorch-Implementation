import cv2 
import numpy as np
from typing import Tuple, Union
from glob import glob 
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import matplotlib.patches as patches

def draw_boundary_box(image_path : str, coordinates : Tuple[Tuple[int]], color : Tuple[int], thickness : int,
                      output_directory : Union[str, None], show : bool = False): 
    """
    Draws multiple boundary boxes on an image with OpenCV 

    Args: 
        image_path (str): direct directory to the image 
        coordinates (Tuple[Tuple[int]]): A tuple of coordinates sets, (xmin, ymin, xmax, ymax) for each boundary box
        color (Tuple[int]): Specification of color with RGB values (R, G, B)
        thickness (int); pixel thickness of boundary box 
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """
    
    image = cv2.imread(image_path)
    
    for box in coordinates:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
    
    if show: 
        cv2.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)    

    return image

def crop_region_of_interest(image_path : str, coordinates : Tuple[int], output_directory : Union[str, None], show : bool = False):
    """
    Crops image based on region of interest coordinates

    Args: 
        image_path (str): direct directory to the image 
        coordinates (Tuple[int]): a set of coordinates, (xmin, ymin, xmax, ymax), representing a single region of interest
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """
    image = cv2.imread(image_path)
    xmin, ymin, xmax, ymax = coordinates
    cropped_image = image[ymin:ymax, xmin:xmax]
    
    if show: 
            cv2.imshow(cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,cropped_image)   

    return cropped_image

def add_gaussian_noise(image_path : str, mean : int, std : int, output_directory : Union[str, None], show : bool = False): 
    """
    Adds Gaussian Noise to a given image with a specified mean and standardeviation 

    Args: 
        image_path (str): direct directory to the image 
        mean (int): mean distribution of gaussian noise 
        std (int): standard deviation of the guassian noise distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """ 
    image = cv2.imread(image_path)

    gaussian_noise = np.zeros(image.shape, dtype = np.uint8)
    
    cv2.randn(gaussian_noise, mean=mean, std=std)
    
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    
    image = cv2.add(image, gaussian_noise)
    
    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def add_uniform_noise(image_path : str, output_directory : Union[str, None], lower_bound : int, upper_bound : int, show : bool = False):
    """
    Adds Uniform Noise to a given image with a specified lower and upper bound. 

    Args: 
        image_path (str): direct directory to the image 
        lower_bound (int): lower bound of the uniform distribution
        upper_bound (int): upper bound of the uniform distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """
    image = cv2.imread(image_path)

    uni_noise = np.zeros(image.shape, dtype = np.unint8)

    cv2.randu(uni_noise, low=lower_bound, high=upper_bound)

    image = cv2.add(image, uni_noise)

    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def add_impulse_noise(image_path : str, output_directory : Union[str, None], lower_bound : int, upper_bound : int, show : bool = False):
    """
    Adds Impulse Noise (Pepper Noise) to a given image.
    
    Args: 
        image_path (str): direct directory to the image 
        lower_bound (int): lower bound of the uniform distribution
        upper_bound (int): upper bound of the uniform distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """
    image = cv2.imread(image_path)

    imp_noise = np.zeros(image.shape, dtype = np.unint8)

    cv2.randu(imp_noise, low=lower_bound, high=upper_bound)
    
    imp_noise = cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]

    image = cv2.add(image, imp_noise)

    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def batch_noise(root_dir : str, output_dir : Union[str, None], show : bool = False, mode : str = "gaussian", **kwargs):
    """
    Applies noise to all images in a specified directory and saves the modified images to an output directory. 
    The function supports different types of noise such as Gaussian, uniform, and impulse.

    Args:
        root_dir (str): The directory containing the images to process.
        output_dir (Union[str, None]): The directory where the noised images will be saved. If None, images are not saved.
        show (bool): If True, displays the noised image. Defaults to False.
        mode (str): The type of noise to apply. Options include 'gaussian', 'uniform', or 'impulse'.
        **kwargs: Keyword arguments specific to the type of noise:
            For 'gaussian':
                mean (float): The mean of the Gaussian noise.
                std (float): The standard deviation of the Gaussian noise.
            For 'uniform' and 'impulse':
                lower_bound (float): The lower bound of the noise distribution.
                upper_bound (float): The upper bound of the noise distribution.

    Raises:
        ValueError: If an invalid mode is specified.
    """
    image_paths = glob(os.path.join(root_dir, "/*"))
    if mode.lower() == "gaussian":           
        for image in tqdm(image_paths): 
            add_gaussian_noise(image_path=image, 
                               output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]), 
                               mean = kwargs['mean'], std=kwargs['std'], 
                               show=False)
    elif mode.lower() == "uniform": 
        for image in tqdm(image_paths): 
            add_uniform_noise(image_path=image, 
                              output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]),
                              lower_bound=kwargs['lower_bound'], upper_bound=kwargs['upper_bound'], 
                              show=False)
    elif mode.lower() == "impulse":
        for image in tqdm(image_paths): 
            add_uniform_noise(image_path=image, 
                              output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]),
                              lower_bound=kwargs['lower_bound'], upper_bound=kwargs['upper_bound'], 
                              show=False)
    else: 
        raise ValueError(f"[Error] Invalid mode. {mode} is not available as a noise mode.")
    
def plot_confusion_matrix(true_labels : np.array, predictions : np.array, num_classes : int, save_pth : Union[str, None]):
    """
    Computes and plots a confusion matrix.
    
    Args:
        true_labels (np.array): 1D NumPy array of true class labels
        predictions (np.array): 1D NumPy array of predicted class labels
        num_classes (np.array): Total number of classes
        save_pth (Union[str, None]): save path for confusion matrix 
    """
    # Compute the confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, predictions):
        cm[true, pred] += 1
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)], 
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    if save_pth: 
        plt.savefig(save_pth)

def visualize_anchors(image, anchors, draw_centers=False):
    """
    Visualizes the anchors and optionally their centers on the given image.

    Args:
        image (torch.Tensor): The input image tensor with shape (channels, height, width).
        anchors (torch.Tensor): The anchors with shape (num_anchors, 4), each row in (x1, y1, x2, y2) format.
        draw_centers (bool): If True, only the centers of the anchors will be drawn.
    """
    if torch.is_tensor(image):
        image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    anchors = anchors.cpu().squeeze(0)

    for i in range(anchors.shape[0]):
        x1, y1, x2, y2 = anchors[i]
        if draw_centers:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ax.plot(center_x, center_y, 'w+')
            print(center_x, center_y)
        else:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()