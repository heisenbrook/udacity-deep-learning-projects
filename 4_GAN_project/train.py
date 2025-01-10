import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from utils.display import display_graph, denormalize
from utils.models import Generator, Discriminator
from utils.preprocess import DatasetDirectory, get_transforms
from utils.optimization import create_optimizers, generator_step, discriminator_step
from tqdm import tqdm
import torch

# you can experiment with different dimensions of latent spaces
latent_dim = 128

# update to cpu if you do not have access to a gpu
if torch.cuda.is_available():
    print('GPU available!')
else:
    print('GPU not available!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number of epochs to train your model
n_epochs = 500

# number of images in each batch
batch_size = 256

# Create optimizers for the discriminator D and generator G
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
g_optimizer, d_optimizer = create_optimizers(generator, discriminator)

data_dir = 'processed_celeba_small/celeba/'
save_dir = 'generated_images/'
dataset = DatasetDirectory(data_dir, get_transforms((64, 64)))

dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True,  
                        drop_last=True,
                        num_workers=6,
                        pin_memory=True)

fixed_latent_vector = torch.randn(4, latent_dim, 1, 1).float().to(device)

losses = []
num_steps = 2

for epoch in tqdm(range(n_epochs), 
                    desc='Epoch', 
                    total=n_epochs,
                    leave=True,
                    ncols=80):
    
    for batch_i, real_images in tqdm(
        enumerate(dataloader),
        desc="Batch",
        total=len(dataloader),
        leave=False,
        ncols=80,
    ):
        generator.train()
        discriminator.train()
        real_images = real_images.to(device)

        z = np.random.uniform(-1, 1, size=(batch_size, latent_dim, 1, 1))
        z = torch.from_numpy(z).float().to(device)

        # update generator every num_steps iterations of critic
        d_loss = discriminator_step(z, real_images, generator, discriminator, d_optimizer)
        if batch_i % num_steps == 0:
            g_loss = generator_step(z, generator, discriminator, g_optimizer)
        
        # append discriminator loss and generator loss
        g = g_loss['loss'].item()
        d = d_loss['loss'].item()
        losses.append((d, g))

    if epoch in np.linspace(1, n_epochs, num=(n_epochs//50), dtype=int):
        generator.eval()
        generated_images = generator(fixed_latent_vector)  
        for i, image in enumerate(generated_images):
            image = image.detach().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image_d = denormalize(image)
            filename = f'Image_{i+1}_epoch_{epoch}.png'
            path = os.path.join(save_dir, filename, cv2.cvtColor(image_d, cv2.COLOR_RGB2BGR))  
            cv2.imwrite(path, image_d)

display_graph(losses)