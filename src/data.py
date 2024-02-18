import torch
from torch.utils.data import DataLoader
import random

class Sampler:
    def __init__(
        self, device='cpu',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
    
    
class DatasetSampler(Sampler):
    def __init__(self, dataset, flag_label, batch_size, num_workers=40, device='cpu'):
        super(DatasetSampler, self).__init__(device=device)
        
        self.loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self.flag_label = flag_label
        
        with torch.no_grad():
            self.dataset = torch.cat(
                [X for (X, y) in self.loader]
                ) if self.flag_label else torch.cat(
                [X for X in self.loader])
 
                
        
    def sample(self, batch_size=8):
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device).float()
        return batch
