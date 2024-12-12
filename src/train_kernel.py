import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from tqdm import tqdm
import sys
 
from src.utils import freeze, unfreeze, make_f_pot, normalize_out_to_0_1, pot
from src.fid_score import get_Z_mmd_pushed_loader_stats, calculate_frechet_distance
from src.cost import kernel_reg

import numpy as np
import matplotlib.pyplot as plt
 


def train_kernel(nets_for_pot,
                nets_for_pot_opt,
                encoder,
                encoder_opt,
                data_samplers,
                generator,
                config):
    
    
    
     
    if config.K > 2 :
        potentials = [make_f_pot(idx, nets_for_pot, config) for idx in range(config.K)]
    else:
        potentials = [ pot(idx,nets_for_pot) for idx in range(config.K)]
    ################################################
    ################################################
        
    
    
    
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
        
        #====================#
        #== inner problem ===#
        #====================#
        
        #====================#
        #  unfreeze encoder  #
        #. freeze potentials #
        #====================#
        
        for k in range(config.K):
            unfreeze(encoder[k])
            
        if config.K > 2:
            for k in range(config.K):
                freeze(nets_for_pot[k])
        else:
            freeze(nets_for_pot[0])
             
        #====================#
        
        for it in tqdm(range(config.INNER_ITERATIONS)):
            
             
            loss_inner = 0 # sum loss for each distribution
            
            #==============================#
            # all encoder params zero grad #
            #==============================#
            encoder_opt.zero_grad() 
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                    for k in range(config.K)]
            
            
            #==============================#
            #    For each distribution     #
            #==============================#
            for k in range(config.K):
                
                #######################
                #=== encoder step ====#
                #######################
                
                Z_cond_k = config.Z_STD * torch.randn(data[k].shape[0],
                                config.RESNET_ENCODER_LATENT,
                                config.ZC,
                                config.IMG_SIZE,
                                config.IMG_SIZE).to(config.DEVICE) #[B,4,1,64,64] 
                
                # checking #
                assert Z_cond_k.requires_grad == False
                 
                
                XZ = torch.cat([ data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1) #[b,4,3,64,64]
                    , Z_cond_k], dim=2) # [B,4,4,64,64]
                
                # checking #
                assert  XZ.shape == torch.Size([data[k].shape[0],config.RESNET_ENCODER_LATENT,
                                              config.ZC +  config.NC,
                                              config.IMG_SIZE,config.IMG_SIZE])
                
                
                XZ = XZ.flatten(start_dim=0, end_dim=1) # [B*4,4,64,64]
                # checking #
                assert XZ.requires_grad == False
                assert XZ.shape == torch.Size([data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                                              config.ZC + config.NC,
                                              config.IMG_SIZE,config.IMG_SIZE])
                 
                
                T_XZ = encoder[k](XZ) #[B*4,512]
                
                 
                
                
                # checking #
                assert T_XZ.requires_grad == True
                assert T_XZ.shape == torch.Size([data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                                                config.LATENT_SIZE])
                
                T_XZ = T_XZ.reshape(data[k].shape[0],config.RESNET_ENCODER_LATENT,
                                    config.LATENT_SIZE) #[B, 4, 512]
                assert T_XZ.requires_grad == True 
                assert T_XZ.shape == torch.Size([data[k].shape[0], 
                                                 config.RESNET_ENCODER_LATENT,
                                                 config.LATENT_SIZE])     
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"sq norm of generated latent by T{k}(x)":
                               torch.norm(T_XZ, dim=2).square().mean().item()}, step=epoch)
                #======================#
                #  encoder step ends   #
                #======================#
                
                
                
                #=== cost computation ===#
                # T_XZ -[B,4,512]
                mapped_x_k = [normalize_out_to_0_1(generator(T_XZ[:,idx,:],
                                    c=None),config) 
                              for idx in range(config.RESNET_ENCODER_LATENT)]
                
                # mapped_x_k[0].shape) : [B,3,64,64]
                
                mapped_x_k = torch.stack(mapped_x_k, dim=1)#[B,4,3,64,64]
                
                # checking # 
                assert mapped_x_k.requires_grad == True
                assert mapped_x_k.shape == torch.Size([data[k].shape[0],
                                                      config.RESNET_ENCODER_LATENT, config.NC,
                                                      config.IMG_SIZE, config.IMG_SIZE])
                
                #  x- [B,3,64,64]; mapped_x_k -[B,4,3,64,64]
                cost_k = 0.5*(data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1).view(
                         data[k].shape[0]*config.RESNET_ENCODER_LATENT,config.NC,
                    config.IMG_SIZE,config.IMG_SIZE) -
                         mapped_x_k.view(data[k].shape[0]*config.RESNET_ENCODER_LATENT,config.NC,
                                         config.IMG_SIZE,config.IMG_SIZE)                     ).flatten(start_dim=1).norm(dim=1,p=2).square().mean(dim=0)
                
                # checking #
                assert cost_k.requires_grad == True
                assert cost_k.shape == torch.Size([])
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"cost of the {k} barycenter":cost_k.item()},
                              step=epoch)
                #========================#
                
                
                
                
                #===== potential ======# 
                potential_k = [potentials[k](normalize_out_to_0_1(generator(T_XZ[:,idx,:],
                                    c=None),config)) for idx in range(config.RESNET_ENCODER_LATENT)]
                
                #flatten
                potential_k = torch.stack(potential_k, dim=1)#[B,4,1]
                assert potential_k.requires_grad == True
                assert potential_k.shape == torch.Size([data[k].shape[0],
                                                       config.RESNET_ENCODER_LATENT,
                                                       1])
                
                potential_k = potential_k.mean(dim=1) #[B,1]
                potential_k = potential_k.mean()
                
                assert potential_k.requires_grad == True
                assert potential_k.shape == torch.Size([])
                
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"potential of the {k} barycenter" : potential_k.item()},
                              step=epoch)
                #======================#
                
                
                
                
                #========================#
                #=========  MMD =========#
                #========================#
                
                # T_XZ : [B,4,512]
                # Z    : [B_z, 512]
                
                
                 
                
                
                #========  MMD 2 =========#
                z_k = torch.randn(data[k].shape[0], config.LATENT_SIZE).to(config.DEVICE) #[B_z,512]
                
                ### gaussian kernel ###
                """
                norm_sq   = torch.cdist(T_XZ.view(data[k].shape[0]*config.RESNET_ENCODER_LATENT
                        , config.LATENT_SIZE), z_k).square() # [B*4,B_z]
                
               
                mmd_k_2 = torch.exp(-.5*norm_sq/config.LATENT_SIZE) # [B*4,B_z]
                """ 
                ### gaussian kernel ###
                
                
                ### distance kernel ###
                
                norm_sq   = torch.cdist(T_XZ.view(data[k].shape[0]*config.RESNET_ENCODER_LATENT
                        , config.LATENT_SIZE), z_k) # [B*4,B_z]
                mmd_k_2 = - norm_sq/config.LATENT_SIZE
                
                ### distance kernel ###
                
                
                mmd_k_2 = mmd_k_2.mean(dim=1)
                mmd_k_2 = -2*mmd_k_2.mean(dim=0) # torch.Size([])
                
                assert  mmd_k_2.requires_grad == True
                assert  mmd_k_2.shape == torch.Size([])
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"mmd_2 of the {k} barycenter": mmd_k_2.item()},
                              step=epoch)
                #=============================#
                
                
                
                #========== MMD_1 ==========#
                idx = torch.triu_indices(config.RESNET_ENCODER_LATENT,
                                         config.RESNET_ENCODER_LATENT,
                                         offset=1)
                
                inside_batch_dist = T_XZ[:,idx[0]]-T_XZ[:,idx[1]] #[B,6,512]
                
                assert inside_batch_dist.requires_grad == True
                assert inside_batch_dist.shape == torch.Size([data[k].shape[0],
                                                             len(idx[0]),config.LATENT_SIZE])
                
                ### gaussian kernel ###
                """ 
                mmd_k_1 = torch.norm(inside_batch_dist,dim=2,p=2).square()/config.LATENT_SIZE #[B,6]
                mmd_k_1 = torch.exp(-.5 * mmd_k_1)#[B,6]
                """
                ### gaussian kernel ###
                
                
                ### distance kernel ###
                
                mmd_k_1 = - torch.norm(inside_batch_dist,dim=2,p=2)/config.LATENT_SIZE
                
                ### distance kernel ###
                
                
                mmd_k_1 = mmd_k_1.mean(dim=1) #[B]
                mmd_k_1 = mmd_k_1.mean(dim=0) #torch.Size([])
                
                assert  mmd_k_1.requires_grad == True
                assert  mmd_k_1.shape == torch.Size([])
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"mmd_1 of the {k} barycenter": mmd_k_1.item()},
                              step=epoch)
                #=============================#
                
                
                
                
                #========= MMD 3 ===========#
                z_s = torch.randn_like(z_k) #[B_z,512]
                idx = torch.triu_indices( z_k.shape[0],
                                         z_k.shape[0],
                                         offset=1)
                
                ### gaussian kernel ###
                """
                mmd_k_3 = torch.norm(z_s[idx[0]] - z_k[idx[1]],dim=1,p=2).square()/config.LATENT_SIZE
                mmd_k_3 = torch.exp(-.5 * mmd_k_3).mean()
                """
                ### gaussian kernel ###
                 
                
                ### distance kernel ###
                mmd_k_3 = - torch.norm(z_s[idx[0]] - z_k[idx[1]],dim=1,p=2)/config.LATENT_SIZE
                
                mmd_k_3 = mmd_k_3.mean(dim=0)
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"mmd_3 of the {k} barycenter": mmd_k_3.item()},
                              step=epoch)
                ### distance kernel ###
                
                
                #===========================#
              
                mmd_sq_k = mmd_k_1 + mmd_k_2 + mmd_k_3
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"mmd_sq of the {k} barycenter":mmd_sq_k.item()},
                              step=epoch)
                #=======================================#
                #=======================================#
                #=======================================#
                
                
                 
                    
                #====== loss-k =====#
                loss_k = config.LAMBDAS[k]*(cost_k - potential_k + config.GAMMA*mmd_sq_k)
                
             
                loss_inner +=  loss_k
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"loss of the {k} barycenter":loss_k.item()},
                              step=epoch)
                #====================#
                
              
            loss_inner.backward()
            encoder_opt.step()
          
        wandb.log({f"loss of inner problem": loss_inner.item()},
                  step=epoch)
        
      
        
        ##############
        ### outer ####
        ##############
        
        
        
        #====================#
        #  freeze encoder     #
        # unfreeze potentials #
        #====================#
        for k in range(config.K):
            freeze(encoder[k])
            
        if config.K > 2:
            
            for k in range(config.K):
                unfreeze(nets_for_pot[k])
        else:
            unfreeze(nets_for_pot[0])
        #=======================#
        
        
        loss_outer = 0
        nets_for_pot_opt.zero_grad()
        data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                    for k in range(config.K)]
            


        for k in range(config.K):


            #=== encoder step ====#
            with torch.no_grad():
                
                Z_cond_k = config.Z_STD * torch.randn(data[k].shape[0],
                                config.RESNET_ENCODER_LATENT,
                                config.ZC,
                                config.IMG_SIZE,
                                config.IMG_SIZE).to(config.DEVICE) #[B,4,1,64,64] 
                assert Z_cond_k.requires_grad == False

                XZ = torch.cat([ data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1) #[b,4,3,64,64]
                    , Z_cond_k], dim=2) # [B,4,4,64,64]
                XZ = XZ.flatten(start_dim=0, end_dim=1) # [B*4,4,64,64]
                assert XZ.requires_grad == False
                assert XZ.shape == torch.Size([data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                                              config.ZC + config.NC,
                                              config.IMG_SIZE,config.IMG_SIZE])

                #unfreeze(encoder)
                T_XZ = encoder[k](XZ) #[B*4,512]
            
                
                assert T_XZ.requires_grad == False
                assert T_XZ.shape == torch.Size([data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                                                config.LATENT_SIZE])

                T_XZ = T_XZ.reshape(data[k].shape[0],config.RESNET_ENCODER_LATENT,
                                    config.LATENT_SIZE) #[B, 4, 512]
                assert T_XZ.requires_grad == False 
                assert T_XZ.shape == torch.Size([data[k].shape[0], 
                                                 config.RESNET_ENCODER_LATENT,
                                                 config.LATENT_SIZE])     
              
            #======================#

            
            #=== cost computation ===#
            # T_XZ -[B,4,512]
            with torch.no_grad():
                mapped_x_k = [normalize_out_to_0_1(generator(T_XZ[:,idx,:],
                                    c=None),config) 
                              for idx in range(config.RESNET_ENCODER_LATENT)]

                # mapped_x_k[0].shape) : [B,3,64,64]

                mapped_x_k = torch.stack(mapped_x_k, dim=1)#[B,4,3,64,64]

                assert mapped_x_k.requires_grad == False
                assert mapped_x_k.shape == torch.Size([data[k].shape[0],
                                                      config.RESNET_ENCODER_LATENT, config.NC,
                                                      config.IMG_SIZE, config.IMG_SIZE])

                #  x- [B,3,64,64]; mapped_x_k -[B,4,3,64,64]:
                cost_k = 0.5*(data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1).view(-1,config.NC,
                                                                                                config.IMG_SIZE,config.IMG_SIZE)-
                                    mapped_x_k.view(-1,config.NC,config.IMG_SIZE,config.IMG_SIZE) ).flatten(start_dim=1).norm(dim=1,p=2).square().mean(dim=0)
                
                #sum(dim=1).mean(dim=0)

                assert cost_k.requires_grad == False
                assert cost_k.shape == torch.Size([])

            #========================#

            #===== potential ======# 
            potential_k = [potentials[k](normalize_out_to_0_1(generator(T_XZ[:,idx,:],
                                c=None),config)) for idx in range(config.RESNET_ENCODER_LATENT)] #flatten
            potential_k = torch.stack(potential_k, dim=1)#[B,4,1]
            assert potential_k.requires_grad == True
            assert potential_k.shape == torch.Size([data[k].shape[0],
                                                   config.RESNET_ENCODER_LATENT,
                                                   1])

            potential_k = potential_k.mean(dim=1) #[B,1]
            potential_k = potential_k.mean()

            assert potential_k.requires_grad == True
            assert potential_k.shape == torch.Size([])
 
            #======================#   

             
            #========  MMD 1 =========#
            with torch.no_grad():
                
                z_k = torch.randn(data[k].shape[0], config.LATENT_SIZE).to(config.DEVICE) #[B_z,512]
                 
                ### gaussian kernel ###
                """
                norm_sq   = torch.cdist(T_XZ.view(data[k].shape[0]*config.RESNET_ENCODER_LATENT
                        , config.LATENT_SIZE), z_k).square() # [B*4,B_z]

                mmd_k_2 = torch.exp(-.5*norm_sq/config.LATENT_SIZE) # [B*4,B_z]
                """
                ### gaussian kernel ###
                
                ### distance kernel ###
               
                norm_sq   = torch.cdist(T_XZ.view(data[k].shape[0]*config.RESNET_ENCODER_LATENT
                        , config.LATENT_SIZE), z_k) # [B*4,B_z]
                mmd_k_2 = - norm_sq/config.LATENT_SIZE
                
                ### distance kernel ###
                
                
                mmd_k_2 = mmd_k_2.mean(dim=1)
                mmd_k_2 = -2*mmd_k_2.mean(dim=0) # torch.Size([])

                assert  mmd_k_2.requires_grad == False
                assert  mmd_k_2.shape == torch.Size([])
               
            #=============================#
            
            
            #========== MMD_2 ==========#
            with torch.no_grad():
                idx = torch.triu_indices(config.RESNET_ENCODER_LATENT,
                                         config.RESNET_ENCODER_LATENT,
                                         offset=1)

                inside_batch_dist = T_XZ[:,idx[0]]-T_XZ[:,idx[1]] #[B,6,512]
                assert inside_batch_dist.requires_grad == False
                assert inside_batch_dist.shape == torch.Size([data[k].shape[0],
                                                            len(idx[0]),config.LATENT_SIZE])
                
                ### gaussian kernel ###
                """
                mmd_k_1 = torch.norm(inside_batch_dist,dim=2,p=2).square()/config.LATENT_SIZE #[B,6]
                mmd_k_1 = torch.exp(-.5 * mmd_k_1)#[B,6]
                """
                ### gaussian kernel ###
                
                ### distance kernel ###
                
                mmd_k_1 = - torch.norm(inside_batch_dist,dim=2,p=2)/config.LATENT_SIZE
                
                ### distance kernel ###
                
                mmd_k_1 = mmd_k_1.mean(dim=1) #[B]
                mmd_k_1 = mmd_k_1.mean(dim=0) #torch.Size([])

                assert  mmd_k_1.requires_grad == False
                assert  mmd_k_1.shape == torch.Size([])
                 
            #=============================#
            
            
            #======= mmd_3 =======#
            z_s = torch.randn_like(z_k) #[B_z,512]
            idx = torch.triu_indices( z_k.shape[0],
                                     z_k.shape[0],
                                     offset=1)
            """
            mmd_k_3 = torch.norm(z_s[idx[0]] - z_k[idx[1]],dim=1,p=2).square()/config.LATENT_SIZE
            mmd_k_3 = torch.exp(-.5 * mmd_k_3).mean()
            """
            mmd_k_3 = - torch.norm(z_s[idx[0]] - z_k[idx[1]],dim=1,p=2)/config.LATENT_SIZE
            mmd_k_3 = mmd_k_3.mean(dim=0)
            #======================#
            
                
            mmd_sq_k = mmd_k_1 + mmd_k_2 + mmd_k_3
            loss_k = config.LAMBDAS[k]*(cost_k - potential_k + config.GAMMA*mmd_sq_k)
            loss_outer +=  loss_k
        
        loss_outer = -1.*loss_outer
        loss_outer.backward()
        nets_for_pot_opt.step()
        
        ##############
        ## plotting ##
        ##############
        
        if epoch % 100 == 0:
            with torch.no_grad():

                data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                        for k in range(config.K)]

                maps = []
                for k in range(config.K):

                    Z_cond_k = config.Z_STD * torch.randn(data[k].shape[0],
                                                config.RESNET_ENCODER_LATENT,
                                                config.ZC,
                                                config.IMG_SIZE,
                                                config.IMG_SIZE).to(config.DEVICE) #[B,4,1,64,64]


                    XZ = torch.cat([ data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1) #[b,4,3,64,64]
                                        , Z_cond_k], dim=2) # [B,4,4,64,64]

                    XZ = XZ.flatten(start_dim=0, end_dim=1) # [B*4,4,64,64]                
                    T_XZ = encoder[k](XZ) #[B*4,512]
                    
                    
                    """ 
                    # sphere
                    norm_t_xz = torch.norm(T_XZ, dim=1, keepdim=True)
                    T_XZ = (np.sqrt(config.LATENT_SIZE)/norm_t_xz)*T_XZ
                    """
                    
                
                    T_XZ = T_XZ.reshape(data[k].shape[0],config.RESNET_ENCODER_LATENT,
                                                        config.LATENT_SIZE) #[B, 4, 512]

                    mapped_x_k = [normalize_out_to_0_1(generator(T_XZ[:,idx,:],c=None),config) 
                                                  for idx in range(config.RESNET_ENCODER_LATENT)]

                    mapped_x_k = torch.stack(mapped_x_k,  dim=1)
                    maps.append(mapped_x_k)

                if config.DATASET != 'mnist':
                    fig,ax = plt.subplots(9,8,figsize=(8,9),dpi=200)

                    for i,k in zip([0,3,6],[0,1,2]):
                        for j in range(8):
                            ax[i,j].imshow(data[k][:8][j].permute(1,2,0).cpu(), cmap = 'gray' if config.DATASET == 'mnist' else None)

                    for j in range(8):
                        for i in range(9):
                            ax[i,j].set_xticks([])
                            ax[i,j].set_yticks([])

                    for k,idx in zip(range(config.K),
                                     [(1,2),(4,5),(7,8)]):
                        for j in range(8):
                            ax[idx[0],j].imshow(maps[k][:8][j][0].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)
                            ax[idx[1],j].imshow(maps[k][:8][j][1].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)

                else:
                    fig,ax = plt.subplots(6,8,figsize=(8,6),dpi=200)
                    
                    for i,k in zip([0,3],[0,1]):
                        for j in range(8):
                            ax[i,j].imshow(data[k][:8][j].permute(1,2,0).cpu(), cmap = 'gray' if config.DATASET == 'mnist' else None)
                    for j in range(8):
                        for i in range(6):
                            ax[i,j].set_xticks([])
                            ax[i,j].set_yticks([])
                            
                    for k,idx in zip(range(config.K),
                                     [(1,2),(4,5)]):
                        for j in range(8):
                            ax[idx[0],j].imshow(maps[k][:8][j][0].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)
                            ax[idx[1],j].imshow(maps[k][:8][j][1].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)
                            
                fig.tight_layout(pad=0.01)
                wandb.log({f"Images":fig},step=epoch)
                plt.show()


def train_kernel_data(nets_for_pot,
                maps,
                nets_for_pot_opt,
                maps_opt,
                data_samplers,
                config):
    
    potentials = [make_f_pot(idx, nets_for_pot, config) for idx in range(config.K)]
    ################################################
    ################################################
    
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
        
        #====================#
        #== inner problem ===#
        #====================#
        
        for k in range(config.K):
            unfreeze(maps[k])
            if k < len(nets_for_pot):
                freeze(nets_for_pot[k])
        
        for it in range(config.INNER_ITERATIONS):
            
             
            loss_inner = 0 # sum loss for each distribution
            
            maps_opt.zero_grad() 
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                    for k in range(config.K)]
            
            
            for k in range(config.K):
                
                #######################
                #== k-th map step ====#
                #######################

                #####################
                # mapping X -> T_XZ
                
                Z_cond_k = config.Z_STD * torch.randn(data[k].shape[0],
                                config.RESNET_ENCODER_LATENT,
                                config.ZC,
                                config.IMG_SIZE,
                                config.IMG_SIZE).to(config.DEVICE) #[B,4,1,64,64] 
                
                # checking #
                assert Z_cond_k.requires_grad == False
                 
                
                XZ = torch.cat([ data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1) #[b,4,3,64,64]
                    , Z_cond_k], dim=2) # [B,4,4,64,64]
                
                # checking #
                assert XZ.requires_grad == False
                assert XZ.shape == torch.Size([data[k].shape[0],config.RESNET_ENCODER_LATENT,
                                              config.ZC +  config.NC,
                                              config.IMG_SIZE,config.IMG_SIZE])
                
                
                XZ = XZ.flatten(start_dim=0, end_dim=1) # [B*4,4,64,64]
                 
                
                mapped_x_k = normalize_out_to_0_1(maps[k](XZ), config)
                    #[B*4,3,64,64]
                # checking #
                assert mapped_x_k.requires_grad == True
                assert mapped_x_k.shape == torch.Size([
                    data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                    config.NC,
                    config.IMG_SIZE,
                    config.IMG_SIZE])

                
                
                ########################
                # cost computation
                
                #  x- [B,3,64,64]; mapped_x_k -[B*4,3,64,64]
                cost_k = 0.5*( 
                    data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1).flatten(start_dim=0, end_dim=1) 
                    -
                    mapped_x_k 
                ).flatten(start_dim=1).norm(dim=1,p=2).square().mean(dim=0)
                
                # checking #
                assert cost_k.requires_grad == True
                assert cost_k.shape == torch.Size([])
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"inner_{k}_cost" : cost_k.item()},
                              step=epoch)

                ######################
                # potentials cost
                
                potential_k = potentials[k](mapped_x_k) # [B*4, 1]
                # checking #
                assert potential_k.requires_grad == True
                assert potential_k.shape == torch.Size([
                    data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                    1])
                
                potential_k = potential_k.mean()
                
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"inner_{k}_pot" : potential_k.item()},
                              step=epoch)
                

                ####################
                # kernel regularization
                
                # mapped_x_k : [B*4, 3, 64, 64]

                kreg = kernel_reg(mapped_x_k.view(
                    data[k].shape[0],
                    config.RESNET_ENCODER_LATENT,
                    config.NC,
                    config.IMG_SIZE,
                    config.IMG_SIZE), config)
              
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"inner_{k}_kreg" : kreg.item()},
                              step=epoch)
                 
                    
                ################
                # final k-th loss

                ##############
                # durty solution to work with large gamma
                _gamma = config.GAMMA
                # if hasattr(config, 'PROGRESSIVE_GAMMA') and config.PROGRESSIVE_GAMMA and config.GAMMA > 10:
                #     if epoch < 200:
                #         _gamma = 10
                #     else:
                #         _gamma = min(config.GAMMA, 10 + (epoch - 200) * (config.GAMMA - 10) / 300)
                # if k == 0 and it == config.INNER_ITERATIONS - 1:
                #     wandb.log({"gamma" : _gamma}, step=epoch)
                loss_k = config.LAMBDAS[k]*(cost_k - potential_k - _gamma*kreg)
                
             
                loss_inner +=  loss_k
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"inner_{k}_loss":loss_k.item()},
                              step=epoch)
                
              
            loss_inner.backward()
            maps_opt.step()
          
        wandb.log({f"inner_loss": loss_inner.item()},
                  step=epoch)
        
        ######################
        ### outer problem ####
        ######################

        for k in range(config.K):
            freeze(maps[k])
            if k < len(nets_for_pot):
                unfreeze(nets_for_pot[k])
        
        
        loss_outer = 0
        nets_for_pot_opt.zero_grad()
        data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                    for k in range(config.K)]
        
        for k in range(config.K):


            ###########
            # mapping 
            
            with torch.no_grad():
                
                Z_cond_k = config.Z_STD * torch.randn(data[k].shape[0],
                                config.RESNET_ENCODER_LATENT,
                                config.ZC,
                                config.IMG_SIZE,
                                config.IMG_SIZE).to(config.DEVICE) #[B,4,1,64,64] 
                assert Z_cond_k.requires_grad == False

                XZ = torch.cat([ data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1) #[b,4,3,64,64]
                    , Z_cond_k], dim=2) # [B,4,4,64,64]
                XZ = XZ.flatten(start_dim=0, end_dim=1) # [B*4,4,64,64]
                assert XZ.requires_grad == False
                assert XZ.shape == torch.Size([data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                                              config.ZC + config.NC,
                                              config.IMG_SIZE,config.IMG_SIZE])

                mapped_x_k = normalize_out_to_0_1(maps[k](XZ), config)
                assert mapped_x_k.shape == torch.Size([
                    data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                    config.NC,
                    config.IMG_SIZE,
                    config.IMG_SIZE])

            ######################
            # potentials cost
            
            potential_k = potentials[k](mapped_x_k) # [B*4, 1]
            # checking #
            assert potential_k.requires_grad == True
            assert potential_k.shape == torch.Size([
                data[k].shape[0]*config.RESNET_ENCODER_LATENT,
                1])
            
            potential_k = potential_k.mean()

            assert potential_k.requires_grad == True
            assert potential_k.shape == torch.Size([])
            loss_outer +=  potential_k
 
        loss_outer.backward()
        nets_for_pot_opt.step()

        wandb.log({f"outer_loss": loss_inner.item()},
                  step=epoch)
        
        ##############
        ## plotting ##
        ##############
        
        if epoch % 100 == 0:
            for k in range(len(nets_for_pot)):
                freeze(nets_for_pot[k])
            
            with torch.no_grad():

                data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE)
                        for k in range(config.K)]

                maps_res = []
                for k in range(config.K):

                    Z_cond_k = config.Z_STD * torch.randn(data[k].shape[0],
                                    config.RESNET_ENCODER_LATENT,
                                    config.ZC,
                                    config.IMG_SIZE,
                                    config.IMG_SIZE).to(config.DEVICE) #[B,4,1,64,64] 
    
                    XZ = torch.cat([ data[k][:,None].repeat(1,config.RESNET_ENCODER_LATENT,1,1,1) #[b,4,3,64,64]
                        , Z_cond_k], dim=2) # [B,4,4,64,64]
                    XZ = XZ.flatten(start_dim=0, end_dim=1) # [B*4,4,64,64]
                    mapped_x_k = normalize_out_to_0_1(maps[k](XZ), config).view(
                        data[k].shape[0],
                        config.RESNET_ENCODER_LATENT,
                        config.NC,
                        config.IMG_SIZE,
                        config.IMG_SIZE)

                    maps_res.append(mapped_x_k)

                if config.DATASET != 'mnist':
                    fig,ax = plt.subplots(9,8,figsize=(8,9),dpi=200)

                    for i,k in zip([0,3,6],[0,1,2]):
                        for j in range(8):
                            ax[i,j].imshow(data[k][:8][j].permute(1,2,0).cpu(), cmap = 'gray' if config.DATASET == 'mnist' else None)

                    for j in range(8):
                        for i in range(9):
                            ax[i,j].set_xticks([])
                            ax[i,j].set_yticks([])

                    for k,idx in zip(range(config.K),
                                     [(1,2),(4,5),(7,8)]):
                        for j in range(8):
                            ax[idx[0],j].imshow(maps_res[k][:8][j][0].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)
                            ax[idx[1],j].imshow(maps_res[k][:8][j][1].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)

                else:
                    fig,ax = plt.subplots(6,8,figsize=(8,6),dpi=200)
                    
                    for i,k in zip([0,3],[0,1]):
                        for j in range(8):
                            ax[i,j].imshow(data[k][:8][j].permute(1,2,0).cpu(), cmap = 'gray' if config.DATASET == 'mnist' else None)
                    for j in range(8):
                        for i in range(6):
                            ax[i,j].set_xticks([])
                            ax[i,j].set_yticks([])
                            
                    for k,idx in zip(range(config.K),
                                     [(1,2),(4,5)]):
                        for j in range(8):
                            ax[idx[0],j].imshow(maps_res[k][:8][j][0].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)
                            ax[idx[1],j].imshow(maps_res[k][:8][j][1].permute(1,2,0).detach().cpu(),
                                               cmap = 'gray' if config.DATASET == 'mnist' else None)
                            
                fig.tight_layout(pad=0.01)
                wandb.log({f"Images":fig},step=epoch)
                plt.show()
                
               
        

