import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import torch.nn as nn
import torch.optim as optim

# Image processing
def load_mnist():
  transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                       std=(0.5, 0.5, 0.5))])

# Download MNIST dataset
  mnist_dataset = datasets.MNIST(root='data', 
                                 train=True, 
                                 transform=transform, 
                                 download=True)
# Split original training set into 70% train and 30% validation
  train_size = int(0.7 * len(mnist_dataset))
  val_size = len(mnist_dataset) - train_size
  train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

# Select a random image from the new training set
  random_index = np.random.randint(len(train_dataset))

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

  mnist_images = train_dataloader.dataset.dataset.data
  mnist_labels = train_dataset.dataset.targets
  return mnist_images, mnist_labels



def load_fashion_mnist():
# Download FashionMNIST dataset
  transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                       std=(0.5, 0.5, 0.5))])
  fashion_mnist_dataset = datasets.FashionMNIST(root='data', 
                                 train=True, 
                                 transform=transform, 
                                 download=True)
# Split original training set into 70% train and 30% validation
  train_size = int(0.7 * len(fashion_mnist_dataset))
  val_size = len(fashion_mnist_dataset) - train_size
  train_dataset, val_dataset = random_split(fashion_mnist_dataset, [train_size, val_size])

# Select a random image from the new training set
  random_index = np.random.randint(len(train_dataset))

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

  fashion_mnist_images = train_dataloader.dataset.dataset.data
  fashion_mnist_labels = train_dataset.dataset.targets
  return fashion_mnist_images, fashion_mnist_labels

def load_cifar():
# Download CIFAR10
  transform = transforms.Compose([
      transforms.ToTensor(),                # Convert to tensor
      transforms.Grayscale(num_output_channels=1),  # Convert RGB to Grayscale
      transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
  ])

  train_dataset = datasets.CIFAR10(root='data', 
                                 train=True, 
                                 transform=transform, 
                                 download=True)

# # Split original training set into 70% train and 30% validation
  train_size = int(0.7 * len(train_dataset))
  val_size = len(train_dataset) - train_size
  train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# # Select a random image from the new training set
# random_index = np.random.randint(len(train_dataset))

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
  cifar_images = torch.tensor(train_dataloader.dataset.dataset.data).permute(0, 3, 1, 2)  # Convert to Tensor & Normalize

# cifar_images = train_dataloader.dataset.dataset.data
  cifar_labels = train_dataset.dataset.targets

  to_grayscale = transforms.Grayscale(num_output_channels=1)
  cifar_gray = torch.stack([to_grayscale(img) for img in cifar_images]) 
  return cifar_images, cifar_labels, cifar_gray

def plot_synthetic_images(synthetic_images, label_idx, set_name = 'cifar-rgb', number_of_rows_and_columns = 10):
    num_bands = synthetic_images.shape[0]
    pixel_size = synthetic_images.shape[-1]
    fig, axs = plt.subplots(number_of_rows_and_columns, number_of_rows_and_columns, figsize=(number_of_rows_and_columns, number_of_rows_and_columns))
    idx = -1
    synthetic_image = synthetic_images[:,0,:,:]
    correction_matrix = torch.zeros(num_bands, pixel_size**2, pixel_size**2)
    mean_vecotr_data = torch.zeros(num_bands, pixel_size**2)
    for idx_i in range(number_of_rows_and_columns):
        for idx_j in range (0, number_of_rows_and_columns):
            idx += 1
            for band_idx in range(num_bands):
                tmp = synthetic_images[band_idx, idx, :, :]
                synthetic_image[band_idx, :, :] = tmp

            if num_bands == 3:
                axs[idx_i,idx_j].imshow(synthetic_image.permute(1,2,0), vmin=0, vmax=1)
            else:
                axs[idx_i,idx_j].imshow(synthetic_image.permute(1,2,0), cmap='gray', vmin=0, vmax=1)

            axs[idx_i,idx_j].set_xticks([])  # Hide x-ticks
            axs[idx_i,idx_j].set_yticks([])  # Hide x-ticks
    fig.suptitle('Fake ' + set_name + ' images')
    plt.savefig('fake-' + set_name + '-class-'+ str(label_idx) +'.pdf')
    plt.close(fig)  # Close figure to free memory

def compute_mean_and_covariance_in_data_space(images, epsilon=1e-6):
    """
    Compute the covariance matrix of the dataset, ensuring it is positive definite.
    """
    images = images.float()  # Ensure floating-point type
    images = images.view(images.shape[0], -1)  # Flatten images
    mean_vector = torch.mean(images, dim=0, keepdim=True)
    centered_images = images - mean_vector
    covariance_matrix = torch.matmul(centered_images.T, centered_images) / (images.shape[0] - 1)
    
    # Regularization: Add a small identity matrix to ensure positive definiteness
    covariance_matrix += epsilon * torch.eye(covariance_matrix.shape[0])
    
    return covariance_matrix, mean_vector

def generate_random_uniform_images_from_gaussian(covariance_matrix_gaussian, num_samples = 1):
    vector_size = covariance_matrix_gaussian.size(0)
    num_pixels = int(vector_size**0.5)
    mean = torch.zeros(vector_size, device=covariance_matrix_gaussian.device)
    
    # Create the multivariate normal distribution
    mvn = torch.distributions.MultivariateNormal(
        loc=mean,
        covariance_matrix=covariance_matrix_gaussian
    )
    
    # Generate samples
    gaussian_samples = mvn.rsample((num_samples,))
    uniform_samples = 0.5 * (1 + torch.erf(gaussian_samples / np.sqrt(2)))  # Convert Gaussian to Uniform [0,1]
    return uniform_samples.reshape(num_samples, num_pixels, num_pixels)


def generate_random_uniform_images(covariance_matrix, num_samples = 1):
    """
    Generate samples from a multivariate normal distribution with mean 0 and given covariance matrix.
    
    Args:
        num_samples (int): Number of samples to generate.
        covariance_matrix (torch.Tensor): Covariance matrix of shape (num_pixels, num_pixels).
        
    Returns:
        torch.Tensor: Tensor of shape (num_samples, num_pixels) containing the samples.
    """
    vector_size = covariance_matrix.size(0)
    num_pixels = int(vector_size**0.5)
    mean = torch.zeros(vector_size, device=covariance_matrix.device)
    
    # Create the multivariate normal distribution
    mvn = torch.distributions.MultivariateNormal(
        loc=mean,
        covariance_matrix=covariance_matrix
    )
    
    # Generate samples
    gaussian_samples = mvn.rsample((num_samples,))
    uniform_samples = 0.5 * (1 + torch.erf(gaussian_samples / np.sqrt(2)))  # Convert Gaussian to Uniform [0,1]
    return uniform_samples.reshape(num_samples, num_pixels, num_pixels)

def compute_gaussian_covariance_from_uniform(covariance_matrix_unifrom, epsilon = 1e-4):
    """
    Compute the Gaussian covariance matrix from the uniform covariance matrix.
    
    Args:
        covariance_matrix_unifrom (torch.Tensor): Covariance matrix of the uniforms, shape (d, d).
    
    Returns:
        torch.Tensor: Gaussian covariance matrix, shape (d, d).
    """
    # Compute Pearson correlation matrix of the uniforms
    diag_var_u = torch.diag(covariance_matrix_unifrom)  # Variances of uniforms (should be ~1/12)
    std_u = torch.sqrt(diag_var_u)    # Standard deviations of uniforms
    outer_std = torch.outer(std_u, std_u)
    R_u = covariance_matrix_unifrom / outer_std         # Pearson correlation matrix of uniforms

    # Compute Gaussian correlation matrix
    R_n = 2 * torch.sin((math.pi / 6) * R_u)

    # Ensure diagonal is exactly 1 (due to numerical precision)
    R_n.fill_diagonal_(1.0)
    covariance_matrix_gaussian =  R_n + torch.eye(R_u.shape[0])
    return covariance_matrix_gaussian

def compute_pixel_cdf_map(images, max_pixel_value = 255):
    """
    Compute the empirical CDF for each pixel position across all images.
    
    Args:
        images (torch.Tensor): Tensor of shape (num_samples, pixel_size, pixel_size),
                               containing grayscale images with integer pixel values (0-255).
    
    Returns:
        torch.Tensor: A tensor of shape (pixel_size**2, 256), where each row contains
                      the empirical CDF for a given pixel position.
    """
    
    num_samples, pixel_size, _ = images.shape
    # max_pixel_value = images.max()
    
    # Flatten images along the pixel dimension
    images_flat = images.view(num_samples, -1)  # Shape: (num_samples, pixel_size**2)
    
    # Initialize the CDF map
    cdf_map = torch.zeros((images_flat.shape[1], max_pixel_value + 1), device=images.device)
    
    # Compute the empirical CDF for each pixel location
    for i in range(images_flat.shape[1]):
        pixel_values = images_flat[:, i]  # All values for a specific pixel position
        hist = torch.bincount(pixel_values, minlength=max_pixel_value + 1).float()  # Count occurrences
        cdf_map[i] = hist.cumsum(dim=0) / num_samples  # Normalize to get CDF
    
    return cdf_map
import torch

def transform_dataset_to_uniform_distribution(images, cdf_map, reverse=False):
    """
    Transform the dataset into a uniform [0,1] dataset using pixel-wise empirical CDF.
    If reverse=True, transform back to the original space.

    Args:
        images (torch.Tensor): Input dataset of shape (num_samples, pixel_size, pixel_size).
                               If reverse=False, dtype=torch.uint8; if reverse=True, dtype=torch.float32.
        cdf_map (torch.Tensor): Empirical CDF map of shape (pixel_size**2, 256).
        reverse (bool): If True, transform back to the original pixel space.

    Returns:
        torch.Tensor: Transformed dataset of the same shape as input.
    """
    num_samples, pixel_size, _ = images.shape
    images_flat = images.view(num_samples, -1)  # Flatten for batch processing
    cdf_map = cdf_map.to(images.device)  # Ensure cdf_map is on the same device

    if not reverse:
        # Forward transformation: Map pixel values to uniform [0,1]
        images_flat = images_flat.long()  # Convert to long for indexing
        transformed_images_flat = cdf_map[torch.arange(images_flat.shape[1], device=images.device).unsqueeze(0), images_flat]
        
        # Reshape and ensure float output
        transformed_images = transformed_images_flat.view(num_samples, pixel_size, pixel_size).to(torch.float32)
        # gaussian_images = torch.erfinv(2 * transformed_images - 1) * torch.sqrt(torch.tensor(2.0))
        return transformed_images#, gaussian_images

    else:
        # Reverse transformation: Map uniform values back to pixel space
        pixel_positions = torch.arange(cdf_map.shape[0], device=images.device)

        # Use `searchsorted` for **each** pixel position separately
        transformed_images_flat = torch.stack([
            torch.searchsorted(cdf_map[i], images_flat[:, i], right=True) for i in range(cdf_map.shape[0])
        ], dim=1).clip(0, 255)  # Shape (num_samples, pixel_size**2)

        # Reshape and ensure byte output
        transformed_images = transformed_images_flat.view(num_samples, pixel_size, pixel_size).to(torch.uint8)
        return transformed_images / 255

def plot_histogram_of_data(data: torch.Tensor, bins: int = 50, color: str = 'blue',
                     title: str = 'Empirical PDF', xlabel: str = 'Value', 
                     ylabel: str = 'Density') -> tuple[plt.Figure, plt.Axes]:
    if type(data) == list and len(data) > 1:
        num_of_plots = len(data)
        fig, ax = plt.subplots(1, num_of_plots, figsize=(5*num_of_plots, 5), tight_layout=True)
        for i in range(num_of_plots):
            data_np = data[i].flatten().cpu().numpy()  # Handles GPU tensors
            ax[i].hist(data_np, bins=bins, density=True, alpha=0.6, color=color, label='Empirical PDF')
            ax[i].axhline(1.0, color='black', linestyle='--', label='Uniform PDF')
            ax[i].set_title(title)
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)
            ax[i].legend()
        return fig, ax
    elif type(data) == list and len(data) == 1:
        data_np = data[0].flatten().cpu().numpy()  # Handles GPU tensors
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
        ax.hist(data_np, bins=bins, density=True, alpha=0.6, color=color, label='Empirical PDF')
        ax.axhline(1.0, color='black', linestyle='--', label='Uniform PDF')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return fig, ax
    elif torch.is_tensor(data):
        data_np = data.flatten().cpu().numpy()  # Handles GPU tensors
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
        ax.hist(data_np, bins=bins, density=True, alpha=0.6, color=color, label='Empirical PDF')
        ax.axhline(1.0, color='black', linestyle='--', label='Uniform PDF')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return fig, ax
    

def min_max_normalization(data):
    data = data
    return (data - data.min()) / (data.max() - data.min())

def transform_data_to_have_the_true_covariance(input_images, covariance_matrix_true, mean_value = 0, ifCholesky = False):

    if not ifCholesky:
        pixel_size = input_images.shape[-1]
        eigenvalues_data, eigenvector_data = torch.linalg.eig(covariance_matrix_true)
        eigenvector_data = eigenvector_data.real
        eigenvalues_data = eigenvalues_data.real
        correction_matrix = torch.matmul(eigenvector_data,torch.diag(eigenvalues_data)**.5)
        # correction_matrix = eigenvector_data.T
        
        # print(input_images.min(), input_images.max())
        input_images = (input_images -  mean_value)
        transformed_images = torch.matmul(correction_matrix.unsqueeze(0), input_images.reshape(-1,pixel_size**2,1)).reshape(-1, pixel_size, pixel_size)

        # print(input_images.min(), input_images.max())
        return transformed_images
    else:
        num_samples, pixel_size, _ = input_images.shape
        p2 = pixel_size ** 2  # Flattened pixel count

        # Flatten images
        input_images_flat = input_images.view(num_samples, p2)


        # Compute empirical covariance of input samples
        mean_input = input_images_flat.mean(dim=0, keepdim=True)
        centered_input = input_images_flat - mean_input
        covariance_input = (centered_input.T @ centered_input) / (num_samples - 1)

        # Compute Cholesky decomposition (or Eigen decomposition)
        L_real = torch.linalg.cholesky(covariance_matrix_true)  # L_real @ L_real.T = covariance_matrix_true
        L_input = torch.linalg.cholesky(covariance_input)  # L_input @ L_input.T = covariance_input

        # Compute correction matrix A
        correction_matrix = L_real @ torch.linalg.inv(L_input)

        # Apply correction matrix
        transformed_images_flat = (input_images_flat - mean_input) @ correction_matrix.T
        transformed_images = transformed_images_flat.view(num_samples, pixel_size, pixel_size)

        return transformed_images



# %%
def correct_fake_images(real_images, fake_images, lr=0.01, num_epochs=500):
    """
    Finds a linear correction matrix A such that:
    Cov(A * fake_images) = Cov(real_images)
    
    Args:
        real_images (torch.Tensor): Tensor of real images of shape (num_samples, pixel_size, pixel_size).
        fake_images (torch.Tensor): Tensor of fake images of shape (num_samples, pixel_size, pixel_size).
        lr (float): Learning rate for gradient descent.
        num_epochs (int): Number of optimization steps.
    
    Returns:
        torch.Tensor: Correction matrix A of shape (pixel_size**2, pixel_size**2).
    """
    lr_first = lr
    num_samples, pixel_size, _ = real_images.shape
    p2 = pixel_size**2  # Number of pixels squared
    
    # Flatten images for matrix operations
    X = real_images.view(num_samples, p2)
    Y = fake_images.view(num_samples, p2)

    # Compute empirical covariance matrices
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    
    Sigma_X = (X - X_mean).T @ (X - X_mean) / (num_samples - 1)
    Sigma_Y = (Y - Y_mean).T @ (Y - Y_mean) / (num_samples - 1)

    # Initialize correction matrix A (identity matrix)
    A = torch.eye(p2, requires_grad=True)

    # Optimizer
    optimizer = optim.Adam([A], lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute transformed covariance
        transformed_cov = A @ Sigma_Y @ A.T
        
        # Loss function: Frobenius norm difference
        loss = torch.norm(Sigma_X - transformed_cov, p='fro')

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Enforce pixel constraints
        with torch.no_grad():
            A.clamp_(0, 255)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}, learning rate = {lr}")
        if loss.item() <= 0.5: lr = lr_first / 100
        elif loss.item() <= 0.8: lr = lr_first / 50
        elif loss.item() <= 1: lr = lr_first / 10
        else: lr = lr_first

    correction_matrix = A.detach()
    corrected_fake_images = torch.matmul(fake_images.view(num_samples, -1), correction_matrix.T).view(num_samples, pixel_size, pixel_size)
    return corrected_fake_images

def correct_fake_images_svd(real_images, fake_images):
    """
    Compute correction matrix A such that Cov(A * fake_images) = Cov(real_images).

    Args:
        real_images (torch.Tensor): (num_samples, pixel_size, pixel_size)
        fake_images (torch.Tensor): (num_samples, pixel_size, pixel_size)

    Returns:
        torch.Tensor: Correction matrix A (pixel_size^2, pixel_size^2)
    """
    num_samples, pixel_size, _ = real_images.shape
    p2 = pixel_size ** 2  # Total pixels per image

    # Flatten images
    real_images_flat = real_images.view(num_samples, -1).float()
    fake_images_flat = fake_images.view(num_samples, -1).float()

    # Compute empirical covariance matrices
    centered_real = real_images_flat - real_images_flat.mean(dim=0, keepdim=True)
    centered_fake = fake_images_flat - fake_images_flat.mean(dim=0, keepdim=True)

    covariance_real = (centered_real.T @ centered_real) / (num_samples - 1)
    covariance_fake = (centered_fake.T @ centered_fake) / (num_samples - 1)

    # SVD decomposition
    U_real, S_real, V_real_T = torch.svd(covariance_real)
    U_fake, S_fake, V_fake_T = torch.svd(covariance_fake)

    # Construct correction matrix A
    sqrt_S_real = torch.diag(torch.sqrt(S_real))
    inv_sqrt_S_fake = torch.diag(1.0 / torch.sqrt(S_fake))

    correction_matrix = U_real @ sqrt_S_real @ V_real_T @ V_fake_T.T @ inv_sqrt_S_fake @ U_fake.T
    
    corrected_fake_images = torch.matmul(fake_images.view(num_samples, -1), correction_matrix.T).view(num_samples, pixel_size, pixel_size)
    return corrected_fake_images

# # Define the Generator (Neural Network)
# class Generator(nn.Module):
#     def __init__(self, pixel_size):
#         super(Generator, self).__init__()
#         input_size = pixel_size**2
#         self.model = nn.Sequential(
#             nn.Linear(input_size, input_size),
#             nn.ReLU(),
#             nn.Linear(input_size, input_size),
#             nn.ReLU(),
#             nn.Linear(input_size, input_size),
#             nn.ReLU(),
#             # nn.Linear(512, 1024),
#             # nn.ReLU(),
#             # nn.Linear(1024, input_size),
#             nn.Tanh()  # Output between [-1, 1]
#         )

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)  # Flatten
#         x = self.model(x)
#         x = (x + 1) / 2   # Rescale to [0, 255]
#         return x.view(-1, pixel_size, pixel_size)  # Reshape to image size

class Generator(nn.Module):
    def __init__(self, pixel_size):
        super(Generator, self).__init__()
        self.pixel_size = pixel_size
        self.model = nn.Sequential(
            nn.ConvTranspose2d(1, 2, kernel_size=4, stride=2, padding=1),  # (B, 64, pixel_size, pixel_size)
            nn.BatchNorm2d(2),
            nn.ReLU(),

            nn.ConvTranspose2d(2, 4, kernel_size=4, stride=2, padding=1),  # (B, 128, 2*pixel_size, 2*pixel_size)
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1),  # (B, 1, 2*pixel_size, 2*pixel_size)
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        x = x.view(-1, 1, self.pixel_size, self.pixel_size)  # Reshape input
        x = self.model(x)
        x = (x + 1) / 2 * 255  # Rescale to [0, 255]
        return x.view(-1, self.pixel_size * 2, self.pixel_size * 2)  # Reshape to imag


# class Generator(nn.Module):
#     def __init__(self, pixel_size):
#         super(Generator, self).__init__()
#         self.pixel_size = pixel_size
#         self.A = nn.Parameter(torch.eye(pixel_size**2))  # Learnable correction matrix

#     def forward(self, x):
#         return torch.matmul(x, self.A.T).view(num_samples, pixel_size, pixel_size)  # Apply correction matrix

# Define the Discriminator (Neural Network)
class Discriminator(nn.Module):
    def __init__(self, pixel_size):
        super(Discriminator, self).__init__()
        input_size = pixel_size**2
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of being real
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten
        return self.model(x)

# Training function
def improve_fake_images_gan(real_images, fake_images, num_epochs=1000, lr=0.0002):
    num_samples, pixel_size, _ = real_images.shape

    # Flatten images
    X = real_images.view(num_samples, -1) / 255  # Normalize to [0,1]
    Y = fake_images.view(num_samples, -1) / 255

    # Initialize models
    G = Generator(pixel_size)
    D = Discriminator(pixel_size)
    

    # Optimizers and Loss Function
    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        # === Train Discriminator ===
        optimizer_D.zero_grad()
        real_labels = torch.ones(num_samples, 1)
        fake_labels = torch.zeros(num_samples, 1)

        real_loss = criterion(D(X), real_labels)
        fake_loss = criterion(D(G(Y).detach()), fake_labels)

        D_loss = real_loss + 10*fake_loss
        D_loss.backward()
        optimizer_D.step()

        # === Train Generator ===
        optimizer_G.zero_grad()
        G_loss = 10*criterion(D(G(Y)), real_labels)  # Fool D
        G_loss.backward()
        optimizer_G.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D_loss = {D_loss.item():.4f}, G_loss = {G_loss.item():.4f}")

    corrected_fake_images = G(Y)#.clamp(0, 255)  # Ensure valid pixel range
    
    return corrected_fake_images.detach()


# %%


def custom_data_loader(uniform_real_images, uniform_fake_images, batch_size, train_split=0.7):
    """
    Creates a DataLoader that yields batches of (real, fake) image pairs.

    Args:
        uniform_real_images (torch.Tensor): Tensor of real uniform images of shape (N, C, H, W) or (N, H, W).
        uniform_fake_images (torch.Tensor): Tensor of fake uniform images of same shape as real images.
        batch_size (int): Batch size for the DataLoader.
        train_split (float): Fraction of data used for training (rest for testing).

    Returns:
        train_loader, test_loader: DataLoaders for training and testing.
    """
    assert uniform_real_images.shape == uniform_fake_images.shape, "Shapes of real and fake images must match"
    
    dataset = TensorDataset(uniform_real_images, uniform_fake_images)
    total_len = len(dataset)
    train_len = int(train_split * total_len)
    test_len = total_len - train_len

    generator = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(dataset, [train_len, test_len], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
