#!/usr/bin/env python
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run_allreduce(rank, size):
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor[0])
    
def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    
if __name__ == "__main__":
    size = 10
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_allreduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
