import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn import Module


def create_optimizers(generator: Module, discriminator: Module):
    """ This function should return the optimizers of the generator and the discriminator """
    
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    return g_optimizer, d_optimizer


# Using Wassersetein distance - better performance and overall results

def generator_loss(fake_logits):
    """ Generator loss, takes the fake scores as inputs. """

    loss = -torch.mean(fake_logits)
    loss.backward()
    return loss

# No use for GP - SN GAN doesn't require that

def discriminator_loss(real_logits, fake_logits):
    """ Discriminator loss, takes the fake and real logits as inputs. """

    r_loss = -torch.mean(real_logits)
    r_loss.backward()
    f_loss = torch.mean(fake_logits)
    f_loss.backward()
    loss = r_loss + f_loss
    return loss


def generator_step(z: torch.Tensor, 
                   generator: Module,
                   discriminator: Module,
                   g_optimizer):
    """ One training step of the generator. """
    # TODO: implement the generator step (foward pass, loss calculation and backward pass)
    
    generator.zero_grad()
    g_optimizer.zero_grad() 

    fake_images = generator(z)

    d_fake = discriminator(fake_images)
    g_loss = generator_loss(d_fake)
        
    g_optimizer.step()

    return {'loss': g_loss}


def discriminator_step(z: torch.Tensor, 
                       real_images: torch.Tensor, 
                       generator: Module, 
                       discriminator: Module, 
                       d_optimizer):
    """ One training step of the discriminator. """
    # TODO: implement the discriminator step (foward pass, loss calculation and backward pass)

    discriminator.zero_grad()
    d_optimizer.zero_grad()

    fake_images = generator(z)

    d_real = discriminator(real_images)
    d_fake = discriminator(fake_images.detach())
    d_loss = discriminator_loss(d_real, d_fake)
    
    d_optimizer.step()
        
    return {'loss': d_loss}