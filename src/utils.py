import torch
import numpy
import matplotlib
from typing import Dict, Any
import os
import pickle
import numpy as np

import wandb
import gc

#-------- initalization for nets -----#
def normalize_out_to_0_1(x, config):
    #assert torch.min(x) < -0.5
    return torch.clip(0.5*(x+1),0,1) if config.DATASET != 'mnist' else torch.clip(x,-1,1)
 
def init_weights(m):
    """Initialization  for MLP"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def weights_init_D(m):
    """Initialization  for ResNet_D"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
#--------------------------------------#        
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    

def clean_resources(*tsrs):
    del tsrs
    gc.collect()
    torch.cuda.empty_cache()
    
#--------------------------------------#    
def make_f_pot(idx, nets, config):
    
    def f_pot(x):
        res = 0.0
        for i, (net, lmbd) in enumerate(zip(nets, config.LAMBDAS)):
            
            if i == idx:
                res += net(x)
            else:
                res -= lmbd * net(x) / (config.K - 1) / config.LAMBDAS[idx]
        return res
    
    return f_pot

def pot(idx, nets_for_pot):
        def res(x):
            mltp =1 if idx == 0 else -1
            return mltp*nets_for_pot[0](x)
        return res
#-------------------------------------#
class Config():

    @staticmethod
    def fromdict(config_dict):
        config = Config()
        for name, val in config_dict.items():
            setattr(config, name, val)
        return config
    
    @staticmethod
    def load(path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'rb') as handle:
            config_dict = pickle.load(handle)
        return Config.fromdict(config_dict)

    def store(self, path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def set_attributes(
            self, 
            attributes_dict: Dict[str, Any], 
            require_presence : bool = True,
            keys_upper: bool = True
        ) -> int:
        _n_set = 0
        for attr, val in attributes_dict.items():
            if keys_upper:
                attr = attr.upper()
            set_this_attribute = True
            if require_presence:
                if not attr in self.__dict__.keys():
                    set_this_attribute = False
            if set_this_attribute:
                if isinstance(val, list):
                    val = tuple(val)
                setattr(self, attr, val)
                _n_set += 1
        return _n_set
    
def get_random_colored_images(images, displace, seed = 0x000000 ):
    np.random.seed(seed)
    
    images = 0.5*(images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.ones(size)-60
    
    for V, H in zip(images, hues - displace):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = torch.stack(colored_images, dim = 0)
  
    
    
    return colored_images

def middle_rgb(x, threshold=0.8):
    
    """
    x - torch.Size([B,3,32,32]) in diapasone [0,1] normalized from Style-GAN and requires_grad
    returns - torch.Size([B,3])
    """
    
    with torch.no_grad():
        # ==== creating_mask ==== 
        x_hsv = matplotlib.colors.rgb_to_hsv(x.permute(0,2,3,1).detach().cpu().numpy()) #[B,32,32,3]
        brightness = torch.from_numpy(x_hsv[:,:,:,-1]) #[B,32,32]
        brightness = brightness.flatten(start_dim=1) #[B, 1024]

        # find indexes with 
        mask = brightness > threshold # [B,1024]
        #==== creating_mask ====
    
    rgb_image = x.flatten(start_dim=2) # [B,3,1024] : requires_grad
    
    # TODO: rewrite in torch form
    output = torch.zeros(rgb_image.shape[0],3).to(x.device)
    for idx,rgb in enumerate(rgb_image): 
        m = mask[idx]
        output[idx][0] = rgb[0][m].mean()
        output[idx][1] = rgb[1][m].mean() 
        output[idx][2] = rgb[2][m].mean() 
    
    return output



def computePotGrad(input, output, create_graph=True, retain_graph=True):
    '''
    :Parameters:
    input : tensor (bs, *shape)
    output: tensor (bs, 1) , NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, *shape)
    '''
    grad = torch.autograd.grad(
        outputs=output, 
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph,
    ) # (bs, *shape) 
    return grad[0]

#----- schedulers ------#
class TrainingSchedulerGeneric:

    @staticmethod
    def extract_kwargs(kwargs, *names, del_names=True):
        vals = []
        for name in names:
            vals.append(kwargs[name])
            if del_names:
                del kwargs[name]
        if len(vals) == 1:
            return vals[0]
        return tuple(vals)

    def __init__(self, *args, **kwargs):
        self._steps_counter = 0

    def on_batch_optim_step(self, epoch=None, batch=None):
        pass

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        self._steps_counter += 1

    def on_epoch_train_end(self, epoch=None):
        gc.collect(); torch.cuda.empty_cache()

    def on_epoch_eval_end(self, epoch=None, losses=None):
        gc.collect(); torch.cuda.empty_cache()


class TrainingSchedulerSM_Mixin(TrainingSchedulerGeneric):

    def __init__(self, *args, **kwargs):
        self.SMscheduler = self.extract_kwargs(kwargs, 'SMscheduler')
        super().__init__(*args, **kwargs)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        super().on_batch_train_end(epoch, batch, losses, data)
        for k, v in losses.items():
            self.SMscheduler.SM.upd("train_{}".format(k), v.item())

    def on_epoch_eval_end(self, epoch=None, losses=None):
        super().on_epoch_eval_end(epoch, losses)
        for k, v in losses.items():
            self.SMscheduler.SM.upd("test_{}".format(k), v)
        self.SMscheduler.epoch()


class TrainingSchedulerWandB_Mixin(TrainingSchedulerGeneric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = self.extract_kwargs(kwargs, 'use_wandb', del_names=False)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        super().on_batch_train_end(epoch, batch, losses, data)
        if self.use_wandb:
            res_dict = {'epoch': epoch}
            for k, v in losses.items():
                res_dict[k] = v
            wandb.log({'train': res_dict}, step=self._steps_counter)

    def on_epoch_eval_end(self, epoch=None, losses=None):
        super().on_epoch_eval_end(epoch, losses)
        if self.use_wandb:
            res_dict = {}
            for k, v in losses.items():
                res_dict[k] = v
            wandb.log({'test': res_dict}, step=self._steps_counter)
        else:
            pass #TODO: add simple print of values


class TrainingSchedulerModelsSaver_Mixin(TrainingSchedulerGeneric):

    def __init__(
        self,
        *args,
        save_models_interval=100,
        rewrite_saved_models=True,
        **kwargs
    ):
        self.model = self.extract_kwargs(
            kwargs, 'model', del_names=False)
        self.save_models_path = self.extract_kwargs(
            kwargs, 'save_models_path')
        self.save_models_interval = save_models_interval
        self.rewrite_saved_models = rewrite_saved_models
        super().__init__(*args, **kwargs)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        if self._steps_counter % self.save_models_interval == 0:
            self.model.store(os.path.join(self.save_models_path, 'model_latest.pth'))
            if not self.rewrite_saved_models:
                file_name = "model_step_{}.pth".format(self._steps_counter)
                self.model.store(os.path.join(self.save_models_path, file_name))
        super().on_batch_train_end(epoch, batch, losses, data)

class TrainingSchedulerFID_IS_Mixin(TrainingSchedulerGeneric):

    def sample_from_model(self, n_samples):
        raise NotImplementedError()

    def __init__(
        self,
        *args,
        save_fid_is_interval=100,
        inception_device='cuda:0',
        inception_batch_size=64,
        estimate_fid_is_n_samples=1000,
        **kwargs
    ):
        self.reference_inception_features_path = self.extract_kwargs(
            kwargs, 'reference_inception_features_path')
        self.compute_fid_is = self.extract_kwargs(kwargs, 'compute_fid_is', del_names=False)
        self.save_fid_is_interval = save_fid_is_interval
        self.inception_device = inception_device
        self.inception_batch_size = inception_batch_size
        self.estimate_fid_is_n_samples = estimate_fid_is_n_samples
        super().__init__(*args, **kwargs)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        if self.compute_fid_is:
            if self._steps_counter % self.save_fid_is_interval == 0:
                ims = self.sample_from_model(self.estimate_fid_is_n_samples)

                (IS_score, IS_score_std), fid_score = get_inception_score_and_fid(
                    ims, self.reference_inception_features_path, 
                    device=self.inception_device, use_torch=False, 
                    batch_size=self.inception_batch_size
                )
                # gc.collect(); torch.cuda.empty_cache()
                losses['IS_score'] = IS_score
                losses['IS_score_std'] = IS_score_std
                losses['FID_score'] = fid_score
                gc.collect(); torch.cuda.empty_cache()

        super().on_batch_train_end(epoch, batch, losses, data)

class TrainingSchedulerLR_Mixin(TrainingSchedulerGeneric):

    def __init__(self, *args, **kwargs):
        self.lr_scheduler = self.extract_kwargs(
            kwargs, 'lr_scheduler', del_names=True)
        super().__init__(*args, **kwargs)

    def on_epoch_eval_end(self, epoch=None, losses=None):
        losses['lr'] = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()
        super().on_epoch_eval_end(epoch, losses)

