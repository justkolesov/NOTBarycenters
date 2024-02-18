import torch
import numpy as np
import matplotlib.pyplot as plt 
import wandb

def plot_intermediate_results(data, mapped, epoch, config, add_label):
    
    if config.DIM == 2:
        
        for k in range(CONFIG.K):
            plt.scatter(data[k][:,0].cpu(),data[k][:,1].cpu(),edgecolor='black',label=f'data {idx+1}')
            plt.scatter(mapped[:,0],mapped[:,1],edgecolor='black',label=f'barycenter {idx+1}')
            plt.grid()
            plt.legend()
            plt.show()
        
    
    elif config.DIM != 2 and config.DATASET =='mnist':
        
        fig,ax = plt.subplots(4,8,figsize=(6,3),dpi=200)
        for i in range(8):
            ax[0,i].imshow(data[0][i].permute(1,2,0).cpu(),cmap='gray')
        for i in range(8):
            ax[2,i].imshow(data[1][i].permute(1,2,0).cpu(),cmap='gray')
        for i in range(8):
            ax[1,i].imshow(mapped[0][i].permute(1,2,0).detach().cpu(),cmap='gray')
        for i in range(8):
            ax[3,i].imshow(mapped[1][i].permute(1,2,0).detach().cpu(),cmap='gray')
        for i in range(4):
            for j in range(8):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
        fig.tight_layout(pad=0.001)
        wandb.log({"Barycenter Images of {distribution}":fig},step=epoch)
        
        
    elif config.DIM != 2 and config.DATASET =='ave_celeba':
        
        fig,ax = plt.subplots(6,8,figsize=(8,6),dpi=200)
        for i in range(8):
            ax[0,i].imshow(data[0][i].permute(1,2,0).cpu() )
        for i in range(8):
            ax[2,i].imshow(data[1][i].permute(1,2,0).cpu() )
        for i in range(8):
            ax[4,i].imshow(data[2][i].permute(1,2,0).cpu() )
        for i in range(8):
            ax[1,i].imshow(mapped[0][i].permute(1,2,0).detach().cpu() )
        for i in range(8):
            ax[3,i].imshow(mapped[1][i].permute(1,2,0).detach().cpu() )
        for i in range(8):
            ax[5,i].imshow(mapped[2][i].permute(1,2,0).detach().cpu() )

        for i in range(6):
            for j in range(8):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
        fig.tight_layout(pad=0.001)
        wandb.log({"Barycenter Images " + add_label + " of distributions":fig},step=epoch)
    
