import torch
import torchvision 

import sys
from src.utils import middle_rgb

def strong_cost(x,y):
    
    """
    x - torch.Size([B,1,H,H])
    y - torch.size([B,1,H,H])
    returns - torch.Size([B,1])
    """
    return torch.norm(x.flatten(start_dim=1,end_dim=-1) - y.flatten(start_dim=1,end_dim=-1),
                      dim=1, keepdim=True).pow(2).mul(0.5)



def cost_image_color_latent(colors_x, y):
    
    """
    colors_x - torch.Size([B,3]) 
    y - torch.Size([B,3,H,W])  - normalize from 0 to 1 
    returns - torch.Size([B,1])
    """
    colors_y = middle_rgb(y) # [B,3]
    
    return torch.norm(colors_x - colors_y,dim=1,p=2,
                      keepdim=True).pow(2).mul(0.5)



def cost_image_shape_latent(x,y):
    
    """
    x - torch.Size([B,1,H,W]) - normalize from 0 to 1
    y - torch.Size([B,3,H,W]) - normalize from 0 to 1
    returns - torch.Size([B,1])
    """
    trnsfm = torchvision.transforms.Grayscale()
    gray_data = trnsfm(y) # [B,1,H,W]
    cost = torch.norm(x.flatten(start_dim=1,end_dim=-1)
                      - gray_data.flatten(start_dim=1,end_dim=-1),p=2,dim=1,keepdim=True).pow(2).mul(0.5)
    
    return cost

def kernel_reg(T_XZ, config):
    # T_XZ: [bs, zdim, 3, 64, 64]
    Z_SIZE = config.RESNET_ENCODER_LATENT
    if config.KREG == 'mse':
        # Weak quadratic cost (normalized by DIM)
        return .5 *T_XZ.var(dim=1).mean()
    if config.KREG == 'energy':
        # Energy-based quadratic cost (for distance-induced kernel)
        return .5 * torch.cdist(T_XZ.flatten(start_dim=2), T_XZ.flatten(start_dim=2)).mean() * Z_SIZE / (Z_SIZE -1)
    if config.KREG == 'gaussian':
        # Gaussian kernel (normalized by DIM)
        idx = torch.triu_indices(Z_SIZE, Z_SIZE, offset=1)
        return 1. - torch.exp(
            -.5*(T_XZ[:,idx[0]]-T_XZ[:,idx[1]]).square().flatten(start_dim=2).mean(dim=2)
        ).mean()
    if config.KREG == 'laplacian':
        # Laplacian kernel (normalized by DIM)
        idx = torch.triu_indices(Z_SIZE, Z_SIZE, offset=1)
        return 1. - torch.exp(
            -(T_XZ[:,idx[0]]-T_XZ[:,idx[1]]).square().flatten(start_dim=2).mean(dim=2).sqrt()
        ).mean()
    raise Exception(f"Unknown kernel reg type '{config.KREG}'!")
 