import torch
import torch.distributions as dist
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import wandb
import sys
from tqdm import tqdm

 
from src.utils import freeze, unfreeze, normalize_out_to_0_1, make_f_pot, pot
from src.cost import strong_cost
from src.fid_score import get_loader_stats, get_pushed_loader_stats, calculate_frechet_distance,get_Z_pushed_loader_stats



def train_entropic( nets_for_pot,
                    nets_for_pot_opt,
                    encoder,
                    encoder_opt,
                    latent_mlp,
                    latent_mlp_opt,
                    data_samplers,
                    generator,
                    config
                    
                    ):
    
    
    if config.K > 2 :
        potentials = [make_f_pot(idx, nets_for_pot, config) for idx in range(config.K)]
    else:
        potentials = [ pot(idx,nets_for_pot) for idx in range(config.K)]
    
    # base loop for training
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
         
        # training ALAE-encoder model  
        # without training OT potentials
        
        # freezing of potentials
        if config.K > 2:
            for k in range(config.K):
                freeze(nets_for_pot[k])
        else:
            freeze(nets_for_pot[0])
                
        # unfreezing decoder
        unfreeze(encoder)
        if config.ALAE:
            unfreeze(latent_mlp)
        
        
        for it in tqdm(range(config.INNER_ITERATIONS)):
            
            loss = 0
            encoder_opt.zero_grad()
            if config.ALAE:
                latent_mlp.zero_grad()
        
                
            # sample data from each distribution
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                    for k in range(config.K)]
            
            for k in range(config.K):
                
                ##======= get_latent_code ==========##
                
                if config.ALAE:
                    
                    x = data[k]
                   
                    needed_resolution = encoder.decoder.layer_to_resolution[-1]
                    while x.shape[2] > needed_resolution:
                        x = F.avg_pool2d(data[k], 2, 2)
                    if x.shape[2] != needed_resolution:
                        x = F.adaptive_avg_pool2d(data[k], (needed_resolution, needed_resolution))
                
                
                    mu_k_z, log_sigma_k_z = encoder.encode(x, 5, 1)
                     
                    # mu.shape = [B, 1, 256], log_sigma.shape = [B]
                    assert mu_k_z.shape == torch.Size([data[k].shape[0],
                                                   1, config.LATENT_ENCODER_SIZE])
                    assert log_sigma_k_z.shape == torch.Size([data[k].shape[0]])
                    
                    mu_k_z = mu_k_z.reshape(-1, config.LATENT_ENCODER_SIZE)
                    log_sigma_k_z = log_sigma_k_z.reshape(-1,1)
                    
                else:
                    
                    mu_k_z, log_sigma_k_z = encoder(data[k]).chunk(2,dim=1)
                    if it == config.INNER_ITERATIONS - 1:
                        wandb.log({f"mu:sq norm of mean generated latent by T{k}(x)": torch.norm(mu_k_z,dim=1).square().mean().item()},
                              step=epoch)
                        wandb.log({f"sigma:sq norm of mean generated latent by T{k}(x)": torch.norm(torch.exp(log_sigma_k_z),dim=1).square().mean().item()},
                              step=epoch)
                    assert mu_k_z.shape == torch.Size([data[k].shape[0], config.LATENT_SIZE])
                    assert log_sigma_k_z.shape == torch.Size([data[k].shape[0], config.LATENT_SIZE])
                 
                assert mu_k_z.requires_grad == True
                assert log_sigma_k_z.requires_grad == True
                
                
                if config.ALAE:
                    mu_k_z = latent_mlp(mu_k_z)
                
                # reparametrization trick
                snd = torch.randn(mu_k_z.shape[0], mu_k_z.shape[-1] ).to(config.DEVICE) # torch.Size([B, latent_size])
                std = torch.exp(log_sigma_k_z).repeat(1,mu_k_z.shape[-1]) if config.ALAE else torch.exp(log_sigma_k_z)
                z_k = mu_k_z +\
                      snd*std 
                
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"sq norm of generated latent by T{k}(x)": torch.norm(z_k,dim=1).square().mean().item()},
                              step=epoch)
                
                
                if config.SPHERE_PROJECTION:
                    norm = torch.norm(z_k, dim=1, keepdim=True)
                    z_k = (np.sqrt(config.LATENT_SIZE)/norm)*z_k
                    if it == config.INNER_ITERATIONS - 1:
                        wandb.log({f"sq norm: T{k}(x) -> sphere proj": torch.norm(z_k,dim=1).square().mean().item()},
                              step=epoch)
                
                
                if config.CHI_PROJECTION:
                    norm = torch.norm(z_k, dim=1, keepdim=True)
                    chi_k = np.random.chisquare( 1, norm.shape[0]).reshape(-1,1) 
                    chi_k = torch.from_numpy(chi_k).to(config.DEVICE).float() 
                    z_k = (torch.sqrt(chi_k)/norm)*z_k
                    if it == config.INNER_ITERATIONS - 1:
                        wandb.log({f"sq norm: T{k}(x) -> chi proj": torch.norm(z_k,dim=1).square().mean().item()},
                              step=epoch)
                    
                    
                assert z_k.requires_grad == True
                assert z_k.shape == torch.Size([mu_k_z.shape[0], config.LATENT_SIZE]) 
                ##========================================## 
                
                
                
                mapped_k = normalize_out_to_0_1(generator(z_k, c=None),config)
                
                ##======= calculation cost ========##
                 
                cost_k = strong_cost(data[k], mapped_k).mean() #[B,1].mean()
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
            if config.ALAE:
                latent_mlp_opt.step()
            
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
        if config.K > 2:
            for k in range(config.K):
                unfreeze(nets_for_pot[k])
        else:
            unfreeze(nets_for_pot[0])
            
        freeze(encoder)
        if config.ALAE:
            freeze(latent_mlp)
            
        loss = 0
        
        for k in range(config.K):
                
            ##======= get_latent_code ==========##
            if config.ALAE:
                
                x = data[k]
                
                needed_resolution = encoder.decoder.layer_to_resolution[-1]
                while x.shape[2] > needed_resolution:
                    x = F.avg_pool2d(data[k], 2, 2)
                if x.shape[2] != needed_resolution:
                    x = F.adaptive_avg_pool2d(data[k], (needed_resolution, needed_resolution))

                with torch.no_grad():
                    mu_k_z, log_sigma_k_z = encoder.encode(x, 5, 1)
                # mu.shape = [B, 1, 256], log_sigma.shape = [B]
                assert mu_k_z.shape == torch.Size([data[k].shape[0],
                                               1, config.LATENT_ENCODER_SIZE])
                assert log_sigma_k_z.shape == torch.Size([data[k].shape[0]])
                mu_k_z = mu_k_z.reshape(-1, config.LATENT_ENCODER_SIZE)
                log_sigma_k_z = log_sigma_k_z.reshape(-1,1)
                
            else:
                
                with torch.no_grad():
                    mu_k_z, log_sigma_k_z = encoder(data[k]).chunk(2,dim=1)
                
                assert mu_k_z.shape == torch.Size([data[k].shape[0], config.LATENT_SIZE])
                assert log_sigma_k_z.shape == torch.Size([data[k].shape[0], config.LATENT_SIZE])
                
             
            assert mu_k_z.requires_grad == False
            assert log_sigma_k_z.requires_grad == False

            if config.ALAE:
                mu_k_z = latent_mlp(mu_k_z)
            
            # reparametrization trick
            std = torch.exp(log_sigma_k_z).repeat(1,mu_k_z.shape[-1]) if config.ALAE else torch.exp(log_sigma_k_z)
            snd = torch.randn(mu_k_z.shape[0], mu_k_z.shape[-1] ).to(config.DEVICE) # torch.Size([B, latent_size])
            z_k = mu_k_z +\
                  snd*std
            
          
                
            if config.SPHERE_PROJECTION:
                norm = torch.norm(z_k, dim=1, keepdim=True)
                z_k = (np.sqrt(config.LATENT_SIZE)/norm)*z_k
                


            if config.CHI_PROJECTION:
                norm = torch.norm(z_k, dim=1, keepdim=True)
                chi_k = np.random.chisquare( 1, norm.shape[0]).reshape(-1,1) 
                chi_k = torch.from_numpy(chi_k).to(config.DEVICE).float() 
                z_k = (torch.sqrt(chi_k)/norm)*z_k
              
            
            assert z_k.requires_grad == False
            assert z_k.shape == torch.Size([mu_k_z.shape[0], config.LATENT_SIZE])

            ##====================================## 

            with torch.no_grad():
                mapped_k = normalize_out_to_0_1(generator(z_k, c=None),config)
            

            ##======= calculation cost ========##
            
            cost_k = strong_cost(data[k], mapped_k).mean() #[B,1].mean()
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
            
            freeze(encoder)
            if config.ALAE:
                freeze(latent_mlp)
            
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                        for k in range(config.K)]
            
            for k in range(config.K):
                
                if not config.ALAE:
                    with torch.no_grad():
                        mu_k_z, log_sigma_k_z = encoder(data[k][:8]).chunk(2, dim=1)
                    std = torch.exp(log_sigma_k_z)
                else:
                    x = data[k][:8]
                    needed_resolution = encoder.decoder.layer_to_resolution[-1]
                    while x.shape[2] > needed_resolution:
                        x = F.avg_pool2d(data[k][:8], 2, 2)
                    if x.shape[2] != needed_resolution:
                        x = F.adaptive_avg_pool2d(data[k][:8], (needed_resolution, needed_resolution))
                    
                    with torch.no_grad():
                        mu_k_z, log_sigma_k_z = encoder.encode(x, 5, 1)
                    mu_k_z = mu_k_z.reshape(-1, config.LATENT_ENCODER_SIZE)
                    log_sigma_k_z = log_sigma_k_z.reshape(-1,1)
                    mu_k_z = latent_mlp(mu_k_z)
                    std = torch.exp(log_sigma_k_z).repeat(1,config.LATENT_SIZE)


                 
                fig,ax = plt.subplots(6,8,figsize=(8,6),dpi=200)
                for idx in range(8):
                    
                  
                                                              
                    ax[0,idx].imshow(data[k][:8][idx].permute(1,2,0).cpu(),
                                     cmap = 'gray' if config.DATASET == 'mnist' else None)
                    
                 
                for run in range(1,6):
                    snd = torch.randn_like(mu_k_z)
                    z_k = mu_k_z + snd*std
                    
                    if config.SPHERE_PROJECTION:
                        norm = torch.norm(z_k, dim=1, keepdim=True)
                        z_k = (np.sqrt(config.LATENT_SIZE)/norm)*z_k



                    if config.CHI_PROJECTION:
                        norm = torch.norm(z_k, dim=1, keepdim=True)
                        chi_k = np.random.chisquare( 1, norm.shape[0]).reshape(-1,1) 
                        chi_k = torch.from_numpy(chi_k).to(config.DEVICE).float() 
                        z_k = (torch.sqrt(chi_k)/norm)*z_k
                    
                    with torch.no_grad():
                        mapped_k  = normalize_out_to_0_1(generator(z_k,c=None),config)#[8,3,64,64]
                        
                    for idx in range(8):
                         
                        ax[run,idx].imshow(mapped_k[idx].detach().permute(1,2,0).cpu(),
                                          cmap = 'gray' if config.DATASET == 'mnist' else None)

                       
                for i in range(6):
                    for j in range(8):
                        ax[i,j].set_xticks([]);ax[i,j].set_yticks([]);
                        
                fig.tight_layout(pad=0.01)
                 
                wandb.log({f"Barycenter Images {k} " + "fixed" + " of distributions":fig},step=epoch)
                plt.show()
             
        