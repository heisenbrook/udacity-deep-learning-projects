import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def display_graph(losses):
    """ helper function to display images during training """
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.show()


def denormalize(images):
    """Transform images from [-1.0, 1.0] to [0, 255] and cast them to uint8."""
    return ((images + 1.) / 2. * 255).astype(np.uint8)

def save_image(generated_images, epoch, save_dir):
    for i, image in enumerate(generated_images):
        image = image.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image_d = denormalize(image)
        image_d = Image.fromarray(image_d, mode='RGB')
        image_d = image_d.resize((256,256),Image.Resampling.LANCZOS)
        filename = f'Image_{i+1}_epoch_{epoch}'
        path = os.path.join(save_dir, filename) 
        image_d.save(path, format='.png')
