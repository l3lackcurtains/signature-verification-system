import cv2
import os
import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate, zoom


def prepare(im_pth, file, augment=False):
    desired_size = 32
    im = cv2.imread(im_pth, 0)

    old_size = im.shape[:2]

    ratio = float(desired_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [225, 225, 225]
    new_im = cv2.copyMakeBorder(im, top, bottom, left,
                                right, cv2.BORDER_CONSTANT, value=color)
    # TODO: adjust threshold based on signature visibility
    _, thresh1 = cv2.threshold(new_im, 140, 255, cv2.THRESH_BINARY)
    
    rotating_angles = [0, 45, 90, 135, 180, 225, 270, 315]

    ret_imgs = []

    for rot_angle in rotating_angles:
      rimg = rotate(thresh1, rot_angle, mode="nearest", reshape=False)
      fileName, fileExtension = os.path.splitext(file)
      cv2.imwrite('./temp/{}-{}.{}'.format(fileName, rot_angle, fileExtension), rimg)
      ret_imgs.append(rimg)

    if augment:
      return ret_imgs
    else:
      return thresh1


def main():
    path = './Dataset/custom2'
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        impath = folder.decode('utf-8')+'/'+file.decode('utf-8')
        prepare(impath, file.decode('utf-8'))


main()
