import torch
import numpy as np
from tqdm import tqdm
import wandb
from src.utils import freeze, unfreeze, make_f_pot, pot, normalize_out_to_0_1
from src.cost import strong_cost
from src.plotting import plot_intermediate_results
from src.fid_score import get_loader_stats, get_pushed_loader_stats, calculate_frechet_distance,get_Z_pushed_loader_stats




def train(nets_for_pot, 
          maps,
          nets_for_pot_opt,
          maps_opt,
          data_samplers,
          generator,
          config):
    
    """
    nets_for_pot - 
    maps - 
    nets_for_pot_opt - 
    maps_opt - 
    config - 
    """
    
    # fixed data for plotting fixed pairs
    """
    data_fixed = [torch.load(f"/trinity/home/a.kolesov/strong_barycenter/stats/ave_celeba/data_fixed_{k}.pth").to(config.DEVICE) 
                  for k in range(config.K)]
    
    """
    #############################################
    ## loading of statistics of target dataset ##
    ##         for FID computation             ##
    #############################################
    """
    mu_data = [None]*config.K
    sigma_data = [None]*config.K
    
    for k in range(config.K):
        mu_data[k] = torch.load("/trinity/home/a.kolesov/data/statistics/celeba_mu.pth")
        sigma_data[k] = torch.load("/trinity/home/a.kolesov/data/statistics/celeba_sigma.pth")
    """
    #############################################
    
    
    
    #############################################
    ##      creating potential functions       ##
    #############################################
    if config.K > 2 :
        potentials = [make_f_pot(idx, nets_for_pot, config) for idx in range(config.K)]
    else:
        potentials = [ pot(idx,nets_for_pot) for idx in range(config.K)]
    #############################################
    
    
    
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
        ###################################
        ## freezing nets for potentials  ##
        ##     unfreezing maps           ##
        ###################################
        
        for idx in range(config.K) :
            unfreeze(maps[idx])
        
        if config.K > 2:
            for idx in range(config.K):
                freeze(nets_for_pot[idx])
        else:
            freeze(nets_for_pot[0])
        ####################################
        
        
        
        #############################
        ##      inner loop         ##
        #############################
        for it in range(config.INNER_ITERATIONS):
            
            # data sampling
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE) 
                    for k in range(config.K)]
            
            maps_opt.zero_grad()
            loss_inner = 0
            
            #######################################
            # for each distribution of barycenter #
            #######################################
            for k in range(config.K):
                
                if config.FLAG_LATENT :
                    
                    
                    
                    z_k = maps[k](data[k])
                    assert z_k.requires_grad == True
                    assert z_k.shape == torch.Size([data[k].shape[0], 
                                                    config.LATENT_SIZE])
            
                    if it == config.INNER_ITERATIONS - 1:
                        wandb.log({f"sq norm of generated latent by T{k}(x)": torch.norm(z_k, dim=1).square().mean().item()},
                                  step=epoch)
                    
                    
                    
                    if config.SPHERE_PROJECTION:
                        norm = torch.norm(z_k,  dim=1, keepdim=True)#[B,1]
                        z_k = (np.sqrt(config.LATENT_SIZE)/norm)*z_k #[B,512]
                        assert z_k.requires_grad == True
                        assert z_k.shape == torch.Size([data[k].shape[0], 
                                                        config.LATENT_SIZE])
                        if it == config.INNER_ITERATIONS -1:
                            wandb.log({f"sq norm: T{k}(x) -> sphere proj": torch.norm(z_k, dim=1).square().mean().item()},
                                  step=epoch)
                     
                    
                    if config.CHI_PROJECTION:
                        norm = torch.norm(z_k, dim=1, keepdim=True)#[B,1]
                        chi_k = np.random.chisquare( 1, norm.shape[0]).reshape(-1,1) #[B,1]
                        chi_k = torch.from_numpy(chi_k).to(config.DEVICE).float() #[B,1]
                        z_k = (torch.sqrt(chi_k)/norm)*z_k
                        assert z_k.requires_grad == True
                        assert z_k.shape == torch.Size([data[k].shape[0], config.LATENT_SIZE])
                        if it == config.INNER_ITERATIONS -1:
                            wandb.log({f"sq norm: T{k}(x) -> chi proj": torch.norm(z_k, dim=1).square().mean().item()},
                                  step=epoch)
                       
                    
                    mapped_x_k = normalize_out_to_0_1(generator(z_k,c=None),config)
                    assert mapped_x_k.requires_grad == True
                     
                     
                else:
                    ###########################
                    ##  L2 cost in L2 space  ##
                    ###########################
                    
                    mapped_x_k = maps[k](data[k]) #[B,C,H,W]
                    assert mapped_x_k.shape == data[k].shape
                    assert mapped_x_k.requires_grad == True
                
                
                #========= cost computation =======#    
                cost_k = strong_cost(data[k],mapped_x_k) #[B,1]
                assert cost_k.requires_grad == True
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"cost of the {k} barycenter": cost_k.mean(dim=0).item()},
                              step=epoch) 
                #========= cost computation =======#
                
                
                
                #========integral of critic =======#
                if config.FLAG_LATENT_CRITIC:
                    potential_k = potentials[k](z_k)
                else:
                    potential_k = potentials[k](mapped_x_k)#[B,1]
                assert potential_k.requires_grad == True
                assert potential_k.shape == torch.Size([data[k].shape[0],1])
                if it == config.INNER_ITERATIONS - 1:
                    wandb.log({f"potential of the {k} barycenter": potential_k.mean(dim=0).item()},
                              step=epoch)
                #========integral of critic =======#
                
                
                #== loss function for k-th dist  ==#
                loss_k = cost_k - potential_k
                loss_k = loss_k.mean(dim=0)
                if it == config.INNER_ITERATIONS -1:
                    wandb.log({f"loss of the {k} barycenter": loss_k.item()}, 
                              step=epoch)
                 
                #==================================#
                
                loss_inner += config.LAMBDAS[k]*loss_k
  
            loss_inner.backward()
            maps_opt.step()
             
   
        wandb.log({f"loss of inner problem": loss_inner.item()}, 
                  step=epoch)
        
        
        
        ############################
        ###### outer problem #######
        ############################
        
        
        ####################################
        ## unfreezing nets for potentials ##
        ##      freezing maps             ##
        ####################################
        
        for idx in range(config.K) :
            freeze(maps[idx])
        
        if config.K > 2:
            for idx in range(config.K):
                unfreeze(nets_for_pot[idx])
        else:
            unfreeze(nets_for_pot[0])
        
         
        nets_for_pot_opt.zero_grad()
        loss_outer = 0
        
        # one step of outer optimization problem #
        for k in range(config.K):
            
            if config.FLAG_LATENT :
             
                with torch.no_grad():
                    z_k = maps[k](data[k])
                assert z_k.requires_grad == False
                assert z_k.shape == torch.Size([data[k].shape[0], 
                                               config.LATENT_SIZE])
 


                if config.SPHERE_PROJECTION:
                    norm = torch.norm(z_k,  dim=1, keepdim=True)#[B,1]
                    z_k = (np.sqrt(config.LATENT_SIZE)/norm)*z_k #[B,512]
                    assert z_k.requires_grad == False
                    assert z_k.shape == torch.Size([data[k].shape[0],
                                                   config.LATENT_SIZE])
                     


                if config.CHI_PROJECTION:
                    norm = torch.norm(z_k, dim=1, keepdim=True)#[B,1]
                    chi_k = np.random.chisquare( 1, norm.shape[0]).reshape(-1,1) #[B,1]
                    chi_k = torch.from_numpy(chi_k).to(config.DEVICE).float() #[B,1]
                    z_k = (torch.sqrt(chi_k)/norm)*z_k
                    assert z_k.requires_grad == False
                    assert z_k.shape == torch.Size([data[k].shape[0], 
                                                   config.LATENT_SIZE])
                   

                with torch.no_grad():
                    mapped_x_k = normalize_out_to_0_1(generator(z_k,c=None),config)
                assert mapped_x_k.requires_grad == False


                 
            else:
                
                with torch.no_grad():
                    mapped_x_k = maps[k](data[k]) #[B,C,H,W]
                assert mapped_x_k.requires_grad == False
                assert mapped_x_k.shape == data[k].shape
            
            
            #========= cost computation ========#
            cost_k = strong_cost(data[k],mapped_x_k) #[B,1]
            assert cost_k.requires_grad == False
            #===================================#
            
            
            #=======integral of potential ======#
            if config.FLAG_LATENT_CRITIC:
                potential_k = potentials[k](z_k)
            else:
                potential_k = potentials[k](mapped_x_k)#[B,1]
            assert potential_k.requires_grad ==  True
            #===================================#  
            
            #=== loss f k-th distribution ===#
            loss_k = cost_k - potential_k #[B,1]
            loss_k = loss_k.mean(dim=0)
            #================================#
            
            
            loss_outer += config.LAMBDAS[k]*loss_k
        
        loss_outer = -1*loss_outer # for maximization
        loss_outer.backward()
        nets_for_pot_opt.step()
      
        
        
        #=============================#
        #========= Plotting ==========#
        #=============================#
        
        ########################
        # plotting UNFIXED part#
        ########################
        
        if epoch % 100 == 0 :
            
            ###########################
            ### plot unfixed images ###
            ###########################
            
            for k in range(config.K):
                freeze(maps[k])
                
                
            data = [data_samplers[k].sample(config.BATCH_SIZE).to(config.DEVICE) 
                    for k in range(config.K)]
            
            if not config.FLAG_LATENT:
                with torch.no_grad():
                    mapped = [maps[k](data[k]).detach().cpu() for k in range(config.K)]
            else:
                mapped = []
                for k in range(config.K):
                 
                    with torch.no_grad():
                        z_k = maps[k](data[k])
                    
                    if config.SPHERE_PROJECTION:
                        norm = torch.norm(z_k,  dim=1, keepdim=True)#[B,1]
                        z_k = (np.sqrt(config.LATENT_SIZE)/norm)*z_k #[B,512]
                        assert z_k.requires_grad == False
                        
                    if config.CHI_PROJECTION:
                        norm = torch.norm(z_k, dim=1, keepdim=True)#[B,1]
                        chi_k = np.random.chisquare( 1, norm.shape[0]).reshape(-1,1) #[B,1]
                        chi_k = torch.from_numpy(chi_k).to(config.DEVICE).float() #[B,1]
                        z_k = (torch.sqrt(chi_k)/norm)*z_k
                        assert z_k.requires_grad == False
                    
                    with torch.no_grad():
                        mapped.append( normalize_out_to_0_1(generator(z_k,c=None),config) )
                   
            plot_intermediate_results(data, mapped, epoch, config, add_label = "unfixed" )
        
        #######################
        # plotting FIXED part #
        #######################
        """
        if epoch % 100 == 0:
            
             
            for k in range(config.K):
                freeze(maps[k])
                
             
            if not config.FLAG_LATENT:
                with torch.no_grad():
                    mapped = [maps[k](data_fixed[k]).detach().cpu() for k in range(config.K)]
            else:
                mapped = []
                for k in range(config.K):
                 
                    with torch.no_grad():
                        z_k = maps[k](data_fixed[k])
                    
                    if config.SPHERE_PROJECTION:
                        norm = torch.norm(z_k,  dim=1, keepdim=True)#[B,1]
                        z_k = (np.sqrt(config.LATENT_SIZE)/norm)*z_k #[B,512]
                        assert z_k.requires_grad == False
                        
                    if config.CHI_PROJECTION:
                        norm = torch.norm(z_k, dim=1, keepdim=True)#[B,1]
                        chi_k = np.random.chisquare( 1, norm.shape[0]).reshape(-1,1) #[B,1]
                        chi_k = torch.from_numpy(chi_k).to(config.DEVICE).float() #[B,1]
                        z_k = (torch.sqrt(chi_k)/norm)*z_k
                        assert z_k.requires_grad == False
                    
                    with torch.no_grad():
                        mapped.append( normalize_out_to_0_1(generator(z_k,c=None),config) )
                   
            plot_intermediate_results(data_fixed, mapped, epoch, config, add_label = "fixed" )
          
        """
        #=============================#
        #============ FID ============#
        #=============================#
        
        """
        # FID calculation
        if epoch % 500 == 0  and epoch > 1 : #and epoch > 1:
            
            for k in range(config.K):
                freeze(maps[k])
                
            for k in range(config.K):
                
                loader_ = data_samplers[k].loader
                
                if not config.FLAG_LATENT:
                    mu_map, sigma_map = get_pushed_loader_stats(maps[k], loader_, batch_size=8, 
                                              verbose=False, device='cuda',
                                               use_downloaded_weights=False)
                else:
                    mu_map, sigma_map = get_Z_pushed_loader_stats(maps[k], loader_, generator=generator, 
                                         config=config, ZC=config.LATENT_SIZE,
                                         Z_STD=1,
                                         batch_size=8, verbose=False,
                                         device=config.DEVICE,
                                         use_downloaded_weights=False)
                
                fid = calculate_frechet_distance(mu_data[k], sigma_data[k],
                                                 mu_map, sigma_map,
                                                 eps=1e-6)
                
                wandb.log({f'FID_{k}_dataset' : fid}, 
                          step=epoch)
                
        """
        
        ################################
        ##          save model.       ##
        ################################
        """
        if epoch % 500 == 0:
            
            if config.FLAG_LATENT:
                
                for k in range(config.K):
                    freeze(maps[k])
                    torch.save(maps[k].cpu().state_dict() ,
                               f"/trinity/home/a.kolesov/strong_barycenter/ckpt/Latent_space/maps/{k}_{config.NAME_EXP}.pth")
                    
                    maps[k] = maps[k].to(config.DEVICE)
                    
                for k in range(config.K):
                    freeze(nets_for_pot[k])
                    torch.save(nets_for_pot[k].cpu().state_dict() ,
                    f"/trinity/home/a.kolesov/strong_barycenter/ckpt/Latent_space/pots/{k}_{config.NAME_EXP}.pth")
                    
                    nets_for_pot[k] = nets_for_pot[k].to(config.DEVICE)
            else:
                for k in range(config.K):
                    freeze(maps[k])
                    torch.save(maps[k].cpu().state_dict() ,
                               f"/trinity/home/a.kolesov/strong_barycenter/ckpt/L2_space/maps/{k}_{config.NAME_EXP}.pth")

                    maps[k] = maps[k].to(config.DEVICE)

                for k in range(config.K):
                    freeze(nets_for_pot[k])
                    torch.save(nets_for_pot[k].cpu().state_dict() ,
                    f"/trinity/home/a.kolesov/strong_barycenter/ckpt/L2_space/pots/{k}_{config.NAME_EXP}.pth")

                    nets_for_pot[k] = nets_for_pot[k].to(config.DEVICE)

            
        """
            
        
        