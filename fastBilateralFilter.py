# author: Deepankar C.

import cv2
import numpy as np
import scipy.stats

import pdb
import matplotlib.pyplot as plt


class BilateralFilter():
  """ Performs bilateral filtering on an input image.
  Args:
    x - input image
    sigma - standard deviation of Gaussian
    
  """
  def __init__(self, sigma_s=None, sigma_r=0.1, 
    downsample=1, not_separable=False, save_loc='./results'):
    # filter paramters
    self.sigma_s = sigma_s
    self.sigma_r = sigma_r
    self.downsample = downsample
    self.use_separable = not not_separable
    self.save_loc = save_loc
    self.eps = 1e-10

  def filter(self, image):
    # initialize sigma_s
    if(self.sigma_s is None):
      self.sigma_s = np.min(image.shape[:2]) * 0.02

    # normalize
    if('int' in str(image.dtype)):
      image = (image / np.amax(image)).astype(np.float32)
    else:
      image = image.astype(np.float32)
    J_img = np.zeros_like(image)

    # image shape
    img_H, img_W = image.shape

    # resize the image
    dsize = (img_W // self.downsample, img_H // self.downsample)
    I = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    # intensity range gaussian
    rkernel = GaussianRange(I, self.sigma_r)

    # image max and min intensity
    Imax = np.amax(I)
    Imin = np.amin(I)
    nb_segments = int((Imax - Imin) // self.sigma_r + 1)
    Idelta = (Imax - Imin) / nb_segments

    # 1D seperable filtering
    if(self.use_separable):
      kernel_X = Gaussian1D(dsize[0], self.sigma_s / self.downsample).kernel
      kernel_Y = Gaussian1D(dsize[1], self.sigma_s / self.downsample).kernel
      for j in range(nb_segments):
        i_j = j * Idelta + Imin

        # bilateral filtering for specific sampled pixel intensity
        G_j = rkernel.apply(i_j)
        K_j = cv2.sepFilter2D(G_j, ddepth=-1, kernelX=kernel_X, kernelY=kernel_Y)
        H_j = G_j * I
        H_star_j = cv2.sepFilter2D(H_j, ddepth=-1, kernelX=kernel_X, kernelY=kernel_Y)
        J_j = H_star_j / (K_j + self.eps)
        J_j = cv2.resize(J_j, dsize=(img_W, img_H), interpolation=cv2.INTER_LINEAR)

        # get pixel intensity interpolation weights
        I_diff = np.abs(image - i_j)
        mask = I_diff < Idelta
        interpolation_weight = mask * (1 - (I_diff / Idelta))

        # construct filtered image
        J_img = J_img + J_j * interpolation_weight

    else:
      # 2D filtering
      kernel_space = Gaussian2D(dsize, self.sigma_s / self.downsample)
      for j in range(nb_segments):
        i_j = j * Idelta + Imin
        
        # bilateral filtering for specific sampled pixel intensity
        G_j = rkernel.apply(i_j)
        K_j = cv2.filter2D(G_j, ddepth=-1, kernel=kernel_space.kernel)
        H_j = G_j * I
        H_star_j = cv2.filter2D(H_j, ddepth=-1, kernel=kernel_space.kernel)
        J_j = H_star_j / (K_j + self.eps)
        J_j = cv2.resize(J_j, dsize=(img_W, img_H), interpolation=cv2.INTER_LINEAR)

        # get pixel internsity interpolation weights
        I_diff = np.abs(I - i_j)
        mask = I_diff < Idelta
        interpolation_weight = mask * (1 - I_diff / Idelta)

        # construct filtered image
        J_img = J_img  + J_j * interpolation_weight
    J_img2 = cv2.bilateralFilter(image, d=-1, 
      sigmaColor=self.sigma_r, sigmaSpace=self.sigma_s)

    im_save = np.concatenate([J_img, J_img2], axis=-1)
    # cv2.imwrite(self.save_loc+'/image_bf.jpg', (im_save * 255).astype(np.float32), [cv2.IMWRITE_JPEG_QUALITY, 95])
    plt.figure(figsize=(8,8))
    plt.subplot(121)
    plt.imshow(J_img, cmap='gray')
    plt.title('Implemented Bilateral Filter')
    plt.subplot(122)
    plt.imshow(J_img2, cmap='gray')
    plt.title('Bilateral Filter using OpenCV')

    return [J_img, J_img2]


class GaussianRange():
  """ Used to generate an intensity range Gaussian.
  Args:
    x - input image
    sigma - standard deviation of Gaussian
    
  """
  def __init__(self, x, sigma):
    self.x = x
    self.sigma = sigma

  def apply(self, px_intensity):
    distance_squared = (self.x - px_intensity)**2
    response = 1 / (np.sqrt(2 * np.pi) * self.sigma) \
      * np.exp(-distance_squared / (2 * self.sigma**2))

    return response


class Gaussian1D():
  """ Used to generate a 1D Gaussian.
  Args:
    ksize - shape of 1D Gaussian
    sigma - standard deviation of Gaussian
    
  """
  def __init__(self, ksize, sigma):
    self.ksize = ksize
    self.sigma = sigma
    self.center = lambda x: x//2 if x//2 == 0 else (x-1)//2
    self.kernel = self.create_gaussian_kernel()

  def create_gaussian_kernel(self):
    gaussian_matrix = np.zeros((self.ksize,))
    values = np.linspace(0, self.ksize, self.ksize, endpoint=False)
    distance_squared = (values - self.center(self.ksize))**2
    self.kernel = 1 / (np.sqrt(2 * np.pi) * self.sigma) \
      * np.exp(-distance_squared / (2 * self.sigma**2))

    return self.kernel


class Gaussian2D():
  """ Used to generate a 2D spatial Gaussian.
  Args:
    ksize - shape of 2D Gaussian
    sigma - standard deviation of Gaussian

  """
  def __init__(self, ksize, sigma):
    self.W, self.H = ksize[0], ksize[1]
    self.sigma = sigma
    self.center = lambda x: x//2 if x//2 == 0 else (x-1)//2
    self.kernel = self.create_spatial_gaussian()

  def create_spatial_gaussian(self):
    gaussian_matrix = np.zeros((self.H, self.W))
    cols, rows = np.meshgrid(
      np.linspace(0, self.H, self.H, endpoint=False), 
      np.linspace(0, self.W, self.W, endpoint=False)
      )

    distance_squared = (cols - self.center(self.W))**2 + (rows - self.center(self.H))**2
    self.kernel = 1 / (2 * np.pi * self.sigma**2) \
      * np.exp(-distance_squared / (2 * self.sigma**2))

    return self.kernel