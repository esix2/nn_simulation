# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import torch.nn as nn
import torch.optim as optim
from utils import *
# %%
def generate_uniform_random_samples_per_class(label_idx, set_name = 'cifar-rgb', epsilon = 0):
    num_bands = 1
    if set_name == 'mnist':
        mnist_images, mnist_labels = load_mnist()
        train_images = mnist_images[mnist_labels == label_idx]
        train_images = train_images.unsqueeze(1)
    elif set_name == 'fashion-mnist':
        fashion_mnist_images, fashion_mnist_labels=  load_fashion_mnist()
        train_images = fashion_mnist_images[fashion_mnist_labels == label_idx]
        train_images = train_images.unsqueeze(1)
    elif set_name == 'cifar-gray':
        _, cifar_labels, cifar_gray = load_cifar()
        train_images = cifar_gray[torch.tensor(cifar_labels) == label_idx,:,:].squeeze(dim=1)
        train_images = train_images.unsqueeze(1)
    elif set_name == 'cifar-rgb':
        num_bands = 3
        cifar_images, cifar_labels, _ = load_cifar()
        train_images = cifar_images[torch.tensor(cifar_labels) == label_idx,:,:,:] 
    else: return [], []
    pixel_size = train_images.shape[-1]
    num_samples = train_images.shape[0]
    train_images = train_images.reshape(num_bands*num_samples, pixel_size, pixel_size)
    pixel_cdf_map = compute_pixel_cdf_map(train_images.reshape(num_bands*num_samples, pixel_size, pixel_size))
    uniform_real_images = transform_dataset_to_uniform_distribution(train_images, pixel_cdf_map, reverse=False)
    # uniform_real_images, _ = transform_dataset_to_uniform_and_gaussian_vectorized(train_images)
    covariance_matrix_real_images, _ = compute_mean_and_covariance_in_data_space(train_images, epsilon=epsilon)
    covariance_matrix_real_uniform, _ = compute_mean_and_covariance_in_data_space(uniform_real_images, epsilon=epsilon)


#     uniform_fake_images = generate_random_uniform_images(covariance_matrix_real_uniform, num_samples = num_samples*num_bands)
    # uniform_fake_images = transform_data_to_have_the_true_covariance(uniform_fake_images.float(), covariance_matrix_real_uniform.float(), mean_value = uniform_fake_images.float().mean(dim=0), ifCholesky=True)
    
    
    covariance_matrix_real_gaussian = torch.eye(pixel_size**2)
    covariance_matrix_real_gaussian = compute_gaussian_covariance_from_uniform(covariance_matrix_real_uniform, epsilon = epsilon)
    covariance_matrix_real_gaussian = covariance_matrix_real_gaussian -0.45*torch.diag(torch.diag(covariance_matrix_real_gaussian))
    uniform_fake_images = generate_random_uniform_images_from_gaussian(covariance_matrix_real_gaussian, num_samples= num_samples*num_bands)

    tmp = uniform_fake_images
    # uniform_fake_images = transform_data_to_have_the_true_covariance(uniform_fake_images.float(), covariance_matrix_real_uniform.float(), mean_value = uniform_fake_images.float().mean(dim=0))
    
    # uniform_fake_images = correct_fake_images(uniform_real_images, uniform_fake_images, num_epochs=1000, lr=0.001)
    # uniform_fake_images = correct_fake_images_svd(uniform_real_images, uniform_fake_images)
#     uniform_fake_images =  improve_fake_images_gan(uniform_real_images, uniform_fake_images, num_epochs=1000, lr=0.00005)
    # uniform_fake_images = min_max_normalization(uniform_fake_images) #/ 255
#     print(uniform_fake_images.min(), uniform_fake_images.max())
    
    
    plot_histogram_of_data([uniform_real_images, tmp,uniform_fake_images])
    # uniform_fake_images = torch.rand(num_bands*num_samples,pixel_size,pixel_size)
    # uniform_fake_images = transform_data_to_have_the_true_covariance(uniform_fake_images, covariance_matrix_real_uniform)
    
    covariance_matrix_fake_uniform, _= compute_mean_and_covariance_in_data_space(uniform_fake_images, epsilon=epsilon)
    print(f"Covarinace difference norm: {(covariance_matrix_real_uniform - covariance_matrix_fake_uniform).norm()}")


    synthetic_images = transform_dataset_to_uniform_distribution(uniform_fake_images, pixel_cdf_map, reverse=True) 
    synthetic_images = synthetic_images.reshape(num_bands, num_samples, pixel_size, pixel_size)

    # synthetic_images = transform_data_to_have_the_true_covariance(synthetic_images.float(), covariance_matrix_real_images.float(), mean_value = synthetic_images.float().mean(dim=0)/5, ifCholesky=False)
#     print(synthetic_images.min(), synthetic_images.max())
    # synthetic_images = min_max_normalization(synthetic_images)


    return uniform_real_images, uniform_fake_images, synthetic_images, pixel_cdf_map

def copula_based_sample_transform():
  save_dir = 'transformed-datasets'
  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
  num_classes = 10
  set_name = 'mnist'
#     set_name = 'fashion-mnist'
#     set_name = 'cifar-rgb'
  set_name = 'cifar-gray'
  ifPlot = True
  output_file = f"{save_dir}/{set_name}-uniform-samples.pth"
  uniform_samples_dict = {}
  for label_idx in range(0,num_classes): 
#   for label_idx in range(7,8):
      print(f"Class: {label_idx}" , end="\r")
      uniform_real_images, uniform_fake_images, synthetic_images, pixel_cdf_map = generate_uniform_random_samples_per_class(label_idx, set_name = set_name, epsilon = 1e-12)
      uniform_samples_dict[label_idx] = (uniform_real_images, uniform_fake_images, pixel_cdf_map)
      plot_synthetic_images(synthetic_images, label_idx, set_name=set_name, number_of_rows_and_columns=5)
  torch.save(uniform_samples_dict, output_file) 
  os.system(f"pdftk fake-{set_name}-class-*.pdf output fake-{set_name}.pdf")
  os.system(f"rm fake-{set_name}*class*")
      # print(tmp1.shape, tmp2.shape)


if __name__ == "__main__":
  copula_based_sample_transform()
