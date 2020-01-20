# authors: Deepankar C. and Jithin Jose Thomas

import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import pdb
%matplotlib inline

from fastBilateralFilter import BilateralFilter

iscolor = True
image = cv2.imread('./images/hdr/tulip.hdr', -1).astype(np.float32)
# image = cv2.cvtColor(cv2.imread('./images/color/3.hdr', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
image = cv2.imread('./images/gray/turtle.ppm', cv2.IMREAD_ANYDEPTH).astype(np.float32)
# image_gray = image

# constants
luminosity_coefficients = np.asarray([20, 40, 1]) / 61
# luminosity_coefficients = np.asarray([0.2126, 0.7152, 0.0722])
sigma_s = np.min(image.shape[:2]) * 0.02
sigma_r = 0.4

eps = 1e-8

# image intensity channel
image_gray = luminosity_coefficients[0] * image[:,:,0] + luminosity_coefficients[1] * image[:,:,1] + luminosity_coefficients[2] * image[:,:,2]
image_gray = (image_gray / np.amax(image_gray))
image_intensity_log = np.log10(image_gray + eps)

# image base and detail scales
bilateralfilter = BilateralFilter(sigma_s, sigma_r, 4)
data = bilateralfilter.filter(image_intensity_log)
image_base = data[0]
# image_base = cv2.bilateralFilter(image_intensity_log, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)
image_detail = image_intensity_log - image_base
image_intensity_BF = 10**(0.5 * image_base + image_detail)

if(iscolor):
  ch_ratio = image_intensity_BF / (image_gray + eps)
  filtered_image = image * ch_ratio[:,:,np.newaxis]
  fim = filtered_image / np.amax(filtered_image)

plt.figure(figsize=(32, 32))
plt.subplot(121)
plt.imshow(image_base, cmap='gray')
plt.subplot(122)
plt.imshow(image_detail, cmap='gray')

plt.figure(figsize=(32,32))
plt.subplot(121)
plt.imshow(image.astype(np.uint8))
plt.subplot(122)
plt.imshow(fim)
