import math
import cv2
import numpy as np
import os
from typing import Union, Tuple


# Function from deskew to calculate rotation angle
def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width_ = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height_ = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width_ - old_width) / 2
    rot_mat[0, 2] += (height_ - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height_)), int(round(width_))), borderValue=background)


# Function to show output
def show_img(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

# Make title for output
def mk_title(org_title, to_join):
    join = str(org_title + str('_' + to_join) + '.jpg')
    return join


# Output Function
def save_img(path, title, image):
    cv2.imwrite(os.path.join(path, title), image)
    return 0
