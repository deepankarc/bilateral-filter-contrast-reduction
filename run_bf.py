# authors: Jithin Jose Thomas & Deepankar C.

import os
import argparse
import timeit

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
    image_color_ori = cv2.imread(args.imagepath, cv2.IMREAD_COLOR)
    image_color = cv2.cvtColor(
      image_color_ori.copy(), 
      cv2.COLOR_BGR2RGB).astype(np.float32)
    image = luminosity_coefficients[0] * image_color[:,:,0] \
      + luminosity_coefficients[1] * image_color[:,:,1] \
        + luminosity_coefficients[2] * image_color[:,:,2]
  else:
    image = cv2.imread(args.imagepath, cv2.IMREAD_ANYDEPTH).astype(np.float32)
  image = (image / np.amax(image))

  if(args.downsample_factor <= 0):
    raise ValueError('The downsampling factor must be positive.')
  
  # start timer
  start = timeit.default_timer()

  # bilateral filter
  bilateralfilter = BilateralFilter(
    args.sigma_s, 
    args.sigma_r, 
    args.downsample_factor, 
    args.not_separable, 
    args.save_loc)
  data = bilateralfilter.filter(image)
  image1, image2 = data[0], data[1]

  # stop timer
  stop = timeit.default_timer()
  print('Bilateral Runtime: ', stop - start) 

  if(args.iscolor):
    ch_ratio = image1 / (image + eps)
    filt_image_color = image_color * ch_ratio[:, :, np.newaxis]
    final_image = filt_image_color / np.amax(filt_image_color)

  # plot images
  fig = plt.figure(figsize=(8,8))
  axes = fig.add_subplot(131)
  axes.imshow(image)
  plt.title('Original Image')
  axes = fig.add_subplot(132)
  axes.imshow(image1)
  plt.title('Image filtered using Bilateral Filter')
  axes = fig.add_subplot(133)
  axes.imshow(image2)
  plt.title('Image filtered using Bilateral Filter from OpenCV')

  # save image
  if(args.iscolor):
    im_save = np.concatenate(
      [image_color_ori / 255, 
      cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)], 
      axis=1)
    cv2.imwrite(
      os.path.join(args.save_loc, fname[:-4]+'.jpg'), 
      (im_save * 255).astype(np.uint8))
  else:
    im_save = np.concatenate([image, image2], axis=1)
    cv2.imwrite(
      os.path.join(args.save_loc, fname[:-4]+'.jpg'), 
      (im_save * 255).astype(np.uint8))


  im_save = np.concatenate([image1, image2], axis=-1)
  cv2.imwrite(os.path.join(args.save_loc, fname[:-4]+'_opencv.jpg'), (im_save * 255).astype(np.uint8))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Bilateral Filtering')
  parser.add_argument('--imagepath', type=str, help='path to the image')
  parser.add_argument('--iscolor', default=False, action='store_true')
  parser.add_argument('--sigma_s', type=float, default=6, help='std of space gaussian')
  parser.add_argument('--sigma_r', type=float, default=0.1, help='std of range gaussian')
  parser.add_argument('--downsample_factor', type=int, default=1)
  parser.add_argument('--not_separable', default=False, action='store_true', help='use full 2D gaussian filter')
  parser.add_argument('--save_loc', default='./results/2019-12-10', help='save to location')

  args = parser.parse_args()
  main(args)
