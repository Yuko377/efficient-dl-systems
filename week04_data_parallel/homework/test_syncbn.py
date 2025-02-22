import torch
from syncbn import SyncBatchNorm
import time
import random
import torch.distributed as dist
import os

def test_batchnorm(num_workers, hid_dim, batch_size):
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    ctx = torch.multiprocessing.get_context("spawn")

    pass

def bn_test(rank, input_tensor, sbn, q):
    # print('World size', dist.get_world_size())
    # print('Rank ', rank, ' has inp ', input_tensor)
    out = sbn(input_tensor)
    print('Rank ', rank, ' has out ', out)

    q.put((out, rank))

    
    

def init_process(rank, size, fn, inp_tensor, sbn, q, master_port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, inp_tensor, sbn, q)

                
if __name__ == "__main__":
    size = 2
    data = torch.Tensor([[1., 1., 1., 1., 1.],
                       [ 0., 0., 0., 0., 0.],
                       [ 0., 0., 0., 0., 0.],
                       [ 0., 0., 0., 0., 0.]])
    # data.requires_grad_()
    processes = []
    sbn = SyncBatchNorm(5)
    port = random.randint(25000, 30000)
    results = []
    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    for rank in range(size):
        p = ctx.Process(target=init_process, args=(rank, size, bn_test, data[rank*2:rank*2 + 2, ], sbn, q, port))
        print('1')
        p.start()
        processes.append(p)
        print(q.get())
        
    for el in results:
        print(el)
    

    for p in processes:
        p.join()
        
    print(data)