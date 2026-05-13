from torch.utils.data.distributed import DistributedSampler
import torch
import math
import numpy as np

        
class FixedSizeDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_samples_per_replica, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=torch.distributed.get_world_size(), 
                         rank=torch.distributed.get_rank(), shuffle=shuffle, seed=seed, drop_last=drop_last)

        self.num_samples_per_replica = num_samples_per_replica
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Padding if not enough
        if len(indices) < self.total_size:
            extra_indices = np.random.choice(indices, self.total_size - len(indices), replace=True).tolist()
            indices += extra_indices
        else:
            indices = indices[:self.total_size]

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_replica
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica