import cv2 
import numpy as np
from typing import Tuple, Union

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
    """ 
    image = cv2.imread(image_path)

    gaussian_noise = np.zeros(image.shape, dtype = np.uint8)
    
    cv2.randn(gaussian_noise, mean=mean, std=std)
    
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    
    gn_img = cv2.add(image, gaussian_noise)
    
    if show: 
            cv2.imshow(gn_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,gn_img)

    return gn_img

