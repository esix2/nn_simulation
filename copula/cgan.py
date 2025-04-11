#!/usr/bin/env python
# coding: utf-8



import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from gen_disc_networks import Generator, Discriminator
# from evaluate_rmse import evaluate_rmse
from IPython.display import Image
# from torchviz import make_dot
# import torch.onnx
from copula import copula_based_sample_transform
from utils import custom_data_loader
import argparse
import logging


def register_hooks(model):
    def hook(module, input, output):
        if module.__class__.__name__.find('Conv') != -1:
#             print(f"{module.__class__.__name__}:")
            print(f"{list(input[0].shape)} --> {list(output.shape)}\n")
    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and layer != model:
            layer.register_forward_hook(hook)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def trainer(num_epochs = 50, batch_size = 64, learning_rate = 1e-3, noise_dim = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = 'checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ifTrain = True
    ifSaveModel = True


    G = Generator(noise_dim)
    D = Discriminator()
#     for param in G.parameters():
#         param.data.fill_(1)
#     for param in D.parameters():
#         param.data.fill_(1)

    generator = G.to(device)
    discriminator = D.to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

#     register_hooks(generator)




    label_idx = 7
#     set_name = 'mnist'
#     set_name = 'fashion-mnist'
#     set_name = 'cifar-rgb'
    set_name = 'cifar-gray'
    saved_uniform_imges_file = f"transformed-datasets/{set_name}-uniform-samples.pth"
    if not os.path.isfile(saved_uniform_imges_file):
        copula_based_sample_transform()
    else: 
        uniform_samples_dict = torch.load(saved_uniform_imges_file)
        uniform_real_images = torch.cat([v[0] for v in uniform_samples_dict.values()], dim=0)
        uniform_fake_images = torch.cat([v[1] for v in uniform_samples_dict.values()], dim=0)
#     uniform_real_images = uniform_real_images[:100]
#     uniform_fake_images = uniform_fake_images[:100]
    train_loader, test_loader = custom_data_loader(uniform_real_images, uniform_fake_images, batch_size=batch_size, train_split=0.9)
    pixel_size = uniform_real_images.shape[-1]

    if ifTrain == True:
        
        # Define loss function and optimizers
        criterion = nn.BCELoss()
        d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.99))
        g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.99))
        
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        for epoch in range(num_epochs):
            for batch_idx, (real_batch, fake_batch) in enumerate(train_loader):
                # Move tensors to device
                fake_batch = fake_batch.unsqueeze(dim=1).to(device)
                real_batch = real_batch.unsqueeze(dim=1).to(device)
        
        
                ############################
                # Train discriminator
                ############################
                d_optimizer.zero_grad()
        
        
        
                # Concatenate the simulated map and measured map along the channel dimension
                real_inputs = torch.cat((fake_batch, real_batch), dim=1)
                real_outputs = discriminator(real_inputs)
                d_loss_real = criterion(real_outputs.squeeze(), real_labels.squeeze())
                d_loss_real.backward()
        
                # Train with fake data
                noise_map = torch.randn(batch_size, noise_dim, pixel_size, pixel_size, device=device)
                fake_images = generator(fake_batch, noise_map)
                fake_inputs = torch.cat((fake_batch, fake_images), dim=1)
                fake_outputs = discriminator(fake_inputs)
                d_loss_fake = criterion(fake_outputs.squeeze(), fake_labels.squeeze())
                d_loss_fake.backward()
        
                d_loss = d_loss_real + d_loss_fake
                d_optimizer.step()
        
                ############################
                # Train generator
                ############################
                g_optimizer.zero_grad()
        
                # Generate fake images
                noise_map = torch.randn(batch_size, noise_dim, pixel_size, pixel_size, device=device)
                fake_images = generator(fake_batch, noise_map)
        
                # Train generator with discriminator feedback
                fake_inputs = torch.cat((fake_batch, fake_images), dim=1)
                outputs = discriminator(fake_inputs)
                g_loss = criterion(outputs.squeeze(), real_labels.squeeze())
                g_loss.backward()
                g_optimizer.step()
        
        
        
                ############################
                # Print losses
                ############################
            if batch_idx % 1 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
            if ifSaveModel == True:
                torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_trained_epoch_{epoch}.pth")
                torch.save(discriminator, f"{save_dir}/discriminator_entire_model_epoch_{epoch}.pth")
                torch.save(generator.state_dict(), f"{save_dir}/generator_trained_epoch_{epoch}.pth")
                torch.save(generator, f"{save_dir}/generator_entire_model_epoch_{epoch}.pth")
#                 print(f"Model saved at epoch {epoch}")


# # Evaluating the results

# ## Showing a sample synthetic map

# In[ ]:


#     generator = Generator(noise_dim).to(device)
#     generator.load_state_dict(torch.load(f"{save_dir}/generator_trained.pth"))
#generator.eval()

#     def generate_image(generator, fake_batch, noise_dim):
#         fake_batch = fake_batch.to(device)
# 
#         noise_map = torch.randn(1, noise_dim, pixel_size, pixel_size, device=device)
#         print("here", fake_batch.shape, noise_map.shape, (fake_batch, noise_map).shape)
# 
#         with torch.no_grad():
#             generated_image = generator(fake_batch, noise_map)
#         return generated_image
# 
#     random_index = random.randint(0, len(test_loader.dataset) - 1)
# 
#     (sample_uniform_real_image_test, sample_uniform_fake_image_test) = test_loader.dataset[random_index]
#     sample_uniform_fake_image_test = sample_uniform_fake_image_test.reshape(1, 1, pixel_size, pixel_size)
# 
# 
#     print(sample_uniform_fake_image_test.shape, fake_batch.shape)
#     sample_uniform_real_image_test_np = sample_uniform_real_image_test.squeeze().numpy()
#     synthetic_image = generate_image(generator, sample_uniform_fake_image_test, noise_dim).numpy()
#     synthetic_image = synthetic_image.squeeze()
#     print(synthetic_image.shape)
# 
# 
#     plt.imshow(sample_uniform_real_images_test_np , cmap='gray')
#     plt.axis('off')
#     plt.show()
# 
#     plt.imshow(sample_uniform_fake_images_test_np , cmap='gray')
#     plt.axis('off')
#     plt.show()
# 
#     plt.imshow(synthetic_image, cmap='gray')
#     plt.axis('off')
#     plt.show()
# 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the copula-based GAN model.")
    parser.add_argument("-b","--batch-size",   type=int, default=64, help="The batch size (default: 64)")
    parser.add_argument("-e","--epochs", type=int, default=50, help="The number of epochs (default: 50)")
    parser.add_argument("-l","--learning-rate",   type=float, default=0.001, help="The learning rate (default: 1e-3)")
    parser.add_argument("-n","--noise-dim",   type=int, default=1, help="The noise input noise dimension (default: 1)")
#     parser.add_argument("-r","--remove-old-output", action="store_true", help="If given, it remove the existing output csv file.")
    args = parser.parse_args()
#     query_the_area (kml_file_path = args.kml_file, date_from = "2020-01-01", date_to = "2024-12-31")
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    learning_rate = args.learning_rate
    noise_dim = args.noise_dim
    trainer(num_epochs, batch_size, learning_rate = learning_rate, noise_dim = noise_dim)
