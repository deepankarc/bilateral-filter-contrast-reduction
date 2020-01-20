# author: Deepankar C.

import os
import argparse

import cv2
import pdb
import numpy as np
import matplotlib.pyplot as plt

from fastBilateralFilter import BilateralFilter

def main(args):
  # useful constants
  eps = 1e-10
  _, fname = os.path.split(args.imagepath)
  luminosity_coefficients = np.asarray([20, 40, 1]) / 61

  # color or B/W and normalize
  if(args.iscolor):
    image_color =  cv2.cvtColor(
      cv2.imread(args.imagepath, cv2.IMREAD_COLOR), 
      cv2.COLOR_BGR2RGB).astype(np.float32)
    image = luminosity_coefficients[0] * image_color[:,:,0] \
      + luminosity_coefficients[1] * image_color[:,:,1] \
        + luminosity_coefficients[2] * image_color[:,:,2]
  else:
    image = cv2.imread(args.imagepath, cv2.IMREAD_ANYDEPTH).astype(np.float32)
  image_gray = (image / np.amax(image))

  if(args.downsample_factor <= 0):
    raise ValueError('The downsampling factor must be positive.')

  # log of intensity
  image_intensity_log = np.log10(image_gray + eps)

  # image base and detail scales
  bilateralfilter = BilateralFilter(
    args.sigma_s, 
    args.sigma_r, 
    args.downsample_factor, 
    args.not_separable, 
    args.save_loc)
  data = bilateralfilter.filter(image_intensity_log)
  image_base = data[0]
  image_detail = image_intensity_log - image_base
  image_intensity_BF = 10**(args.gamma * image_base + image_detail)

  if(args.iscolor):
    ch_ratio = image1 / (image + eps)
    filt_image_color = image_color * ch_ratio[:, :, np.newaxis]
    final_image = filt_image_color / np.amax(filt_image_color)
  else:
    final_image = image_intensity_BF

  # plot images
  fig = plt.figure(figsize=(8,8))
  axes = fig.add_subplot(121)
  axes.imshow(image)
  plt.title('Original Image')
  axes = fig.add_subplot(122)
  axes.imshow(final_image)
  plt.title('Contrast Enhanced using Bilateral Filter')
  plt.show()

  # save image
  im_save = np.concatenate([image_base, image_detail], axis=-1)
  cv2.imwrite(os.path.join(args.save_loc, fname[:-4]+'_basedetail.jpg'), (im_save * 255).astype(np.uint8))

  im_save = np.concatenate([image_gray, final_image], axis=-1)
  cv2.imwrite(os.path.join(args.save_loc, fname[:-4]+'.jpg'), (im_save * 255).astype(np.uint8))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Bilateral Filtering')
  parser.add_argument('--imagepath', type=str, help='path to the image')
  parser.add_argument('--iscolor', default=False, action='store_true')
  parser.add_argument('--sigma_s', type=float, default=None, help='std of space gaussian')
  parser.add_argument('--sigma_r', type=float, default=0.4, help='std of range gaussian')
  parser.add_argument('--gamma', type=float, default=0.5, help='std of range gaussian')
  parser.add_argument('--downsample_factor', type=int, default=1)
  parser.add_argument('--not_separable', default=False, action='store_true', help='use full 2D gaussian filter')
  parser.add_argument('--save_loc', default='./results/2019-12-10', help='save to location')

  args = parser.parse_args()
  main(args)
