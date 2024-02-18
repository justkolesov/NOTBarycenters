import torch
import torch.distributions as dist
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import wandb
import sys
from tqdm import tqdm
 
from src.utils import freeze, unfreeze, normalize_out_to_0_1, make_f_pot, pot, middle_rgb
from src.cost import strong_cost, cost_image_shape_latent, cost_image_color_latent
from src.fid_score import get_loader_stats, get_pushed_loader_stats, calculate_frechet_distance,get_Z_pushed_loader_stats



def train_shape_color_entropic( nets_for_pot,
                    nets_for_pot_opt,
                    encoder,
                    encoder_opt,
                    latent_mlp,
                    latent_mlp_opt,
                    data_samplers,
                    generator,
                    config
                    
                    ):
    
    """
    """
    
    
    
    # potentials
    if config.K == 2 :
        potentials = [ pot(idx,nets_for_pot) for idx in range(config.K)]
    
 
    # base loop for training
    for epoch in tqdm(range(config.NUM_EPOCHS)):
       
        # freezing of potentials
        if config.K == 2:
            freeze(nets_for_pot[0])       
        # unfreezing decoder
        for k in range(config.K):
            unfreeze(encoder[k])
       
        
        for it in tqdm(range(config.INNER_ITERATIONS)):
            
            loss = 0
            encoder_opt.zero_grad()
   
            # sample data from each distribution
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                    for k in range(config.K)]
            
            for k in range(config.K):
                     
            
                mu_k_z, log_sigma_k_z = encoder[k](data[k]).chunk(2,dim=1)
                
                if it == config.INNER_ITERATIONS - 1:
                    
                        wandb.log({f"mu:sq norm of mean generated latent by T{k}(x)": torch.norm(mu_k_z,dim=1).square().mean().item()},
                              step=epoch)
                        
                        wandb.log({f"sigma:sq norm of mean generated latent by T{k}(x)": torch.norm(torch.exp(log_sigma_k_z),dim=1).square().mean().item()},
                              step=epoch)
                     
                assert mu_k_z.requires_grad == True
                assert log_sigma_k_z.requires_grad == True
                
             
                # reparametrization trick
                snd = torch.randn(mu_k_z.shape[0], mu_k_z.shape[-1] ).to(config.DEVICE) # torch.Size([B, latent_size])
                std = torch.exp(log_sigma_k_z)
                z_k = mu_k_z +\
                      snd*std 
                
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"sq norm of generated latent by T{k}(x)": torch.norm(z_k,dim=1).square().mean().item()},
                              step=epoch)
                 
                assert z_k.requires_grad == True
                 ##========================================## 
                
             
                mapped_k = normalize_out_to_0_1(generator(z_k, c=None),config)
                
                ##======= calculation cost ========##
                
                if k == 0:
                    cost_k =  cost_image_shape_latent(data[k], mapped_k).mean() #[B,1].mean()
                elif k == 1:
                    cost_k = 100*cost_image_color_latent(data[k], mapped_k ).mean() #[B,1].mean()
                else:
                    raise ValueError
                    
                    
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"cost of the {k} barycenter": cost_k.item()},
                              step=epoch)
                ##=================================##
                
                
                
                
                ##====== integral of potential  ======##
                
                pot_k = potentials[k](mapped_k).mean()  # [B,1].mean()
                if it == config.INNER_ITERATIONS-1:
                    wandb.log({f"potential of the {k} barycenter": pot_k.item()},
                              step=epoch)
                ##===================================##
                
                
                
                
                
                ##====== calculation entropy_reg =====##
                
                log_std_1 = log_sigma_k_z
                log_std_2 = torch.zeros_like(log_std_1)
                mean_1 = mu_k_z
                mean_2 = torch.zeros_like(mu_k_z)

                kl_div_k = (
                        + torch.exp(2 * (log_std_1 - log_std_2))

                        + ((mean_2 - mean_1) / torch.exp(log_std_2))**2

                        + 2 * (log_std_2 - log_std_1)
                        - 1
                    ) / 2
                
                kl_div_k = torch.mean(kl_div_k.sum(dim=1))
                
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"KL div of distribution {k}": kl_div_k.item()},
                              step=epoch)
                
                ##====================================##
                
                
                
                loss_k = config.LAMBDAS[k]*(cost_k  +  config.EPSILON*kl_div_k - pot_k)
                loss +=  loss_k
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"loss of the {k} barycenter":loss_k.item()},
                              step=epoch)
                
               
            loss.backward()
            encoder_opt.step()
           
        wandb.log({f"loss of inner problem": loss.item()},
                  step=epoch)
                
        #================================#
        #===========  Outer  ============#
        #================================#
        
        
        
        
        
        # Outer optimization problem
        # training OT potential 
        # without training of ALAE encoder
        
        nets_for_pot_opt.zero_grad()
        # unfreezing of potentials 
        if config.K == 2:
            unfreeze(nets_for_pot[0])
         
            
        for k in range(config.K):
            freeze(encoder[k])
       
            
        loss = 0
        
        for k in range(config.K):
                
            ##======= get_latent_code ==========##
             
        

            with torch.no_grad():
                mu_k_z, log_sigma_k_z = encoder[k](data[k]).chunk(2,dim=1)

            
             
            assert mu_k_z.requires_grad == False
            assert log_sigma_k_z.requires_grad == False

          
            
            # reparametrization trick
            std = torch.exp(log_sigma_k_z)
            snd = torch.randn(mu_k_z.shape[0], mu_k_z.shape[-1] ).to(config.DEVICE) # torch.Size([B, latent_size])
            z_k = mu_k_z +\
                  snd*std
        
            assert z_k.requires_grad == False
             

            ##====================================## 

            with torch.no_grad():
                mapped_k = normalize_out_to_0_1(generator(z_k, c=None),config)
            

            ##======= calculation cost ========##
            
            if k == 0:
                cost_k = cost_image_shape_latent(data[k], mapped_k).mean() #[B,1].mean()
            elif k ==1:
                cost_k = 100*cost_image_color_latent(data[k], mapped_k).mean()
            else:
                raise ValueError
                
            assert cost_k.requires_grad == False
            ##=================================##


            ##====== integral of potential  ======##

            pot_k = potentials[k](mapped_k).mean()  # [B,1].mean()
            assert pot_k.requires_grad == True
            ##===================================##



            ##====== calculation entropy_reg =====##
          
            log_std_1 = log_sigma_k_z
            log_std_2 = torch.zeros_like(log_std_1)
            mean_1 = mu_k_z
            mean_2 = torch.zeros_like(mu_k_z)
            
            kl_div_k = (
                    + torch.exp(2 * (log_std_1 - log_std_2))
                
                    + ((mean_2 - mean_1) / torch.exp(log_std_2))**2
                
                    + 2 * (log_std_2 - log_std_1)
                    - 1
                ) / 2
            
           
            kl_div_k = torch.mean(kl_div_k.sum(dim=1))
            assert kl_div_k.requires_grad == False

            ##====================================##



            loss_k = config.LAMBDAS[k]*(cost_k + config.EPSILON*kl_div_k - pot_k)
            loss +=  loss_k
            

        loss = -1*loss 
        loss.backward()
        nets_for_pot_opt.step()
        
        
        
        ##===== plotting results =====##
        
        if epoch % 100 == 0:
            
            for k in range(config.K):
                freeze(encoder[k])
            
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                        for k in range(config.K)]
            
            for k in range(config.K):
                
                with torch.no_grad():
                    mu_k_z, log_sigma_k_z = encoder[k](data[k][:8]).chunk(2, dim=1)
                std = torch.exp(log_sigma_k_z)
       
                fig,ax = plt.subplots(7,8,figsize=(8,7),dpi=200)
        
                for idx in range(8):
                     
                    if k == 0:
                        ax[0,idx].imshow(data[k][:8][idx].permute(1,2,0).cpu(),
                                     cmap = 'gray' if k==0 else None)
                    if k == 1:
                        for idx in range(8):
                            ax[0,idx].set_aspect( 1 ) 
                            ax[0,idx].add_artist(plt.Circle(( 0.5 , 0.5 ), 0.4 ,color=data[k][:8][idx].cpu().numpy() ) ) 
                            ax[0,idx].set_xticks([]);ax[0,idx].set_yticks([]);
                    
                 
                for run in range(1,6):
                    snd = torch.randn_like(mu_k_z)
                    z_k = mu_k_z + snd*std
 
                    with torch.no_grad():
                        mapped_k  = normalize_out_to_0_1(generator(z_k,c=None),
                                                         config)#[8,3,64,64]
                        
                    for idx in range(8):
                         
                        ax[run,idx].imshow(mapped_k[idx].detach().permute(1,2,0).cpu())
   

                clr = middle_rgb(mapped_k,  config.SATURATION_THRESHOLD )
                for idx in range(8):
        
                    if k == 0:
                        ax[6,idx].imshow(data[k][:8][idx].permute(1,2,0).cpu(),
                                     cmap = 'gray' if k==0 else None)
                    if k == 1:
                        ax[6,idx].set_aspect( 1 ) 
                        ax[6,idx].add_artist(plt.Circle(( 0.5 , 0.5 ), 0.4 ,color=clr[idx].cpu().numpy() ) ) 
                        ax[6,idx].set_xticks([]);ax[0,idx].set_yticks([]);
        
        
                for i in range(7):
                    for j in range(8):
                        ax[i,j].set_xticks([]);ax[i,j].set_yticks([]);
                        
                fig.tight_layout(pad=0.01)
                 
                wandb.log({f"Barycenter Images {k} " + "unfixed" + " of distributions":fig},step=epoch)
                plt.show()
                
            
                
            
         
                
   
            
