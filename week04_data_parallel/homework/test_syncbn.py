import torch
from syncbn import SyncBatchNorm
import pytest
import random
import torch.distributed as dist
import os


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    size = num_workers
    feature_num = hid_dim
    torch.manual_seed(1337)
    data = torch.randn(batch_size, feature_num)
    data.requires_grad = True
    bn = torch.nn.BatchNorm1d(feature_num, affine=False, track_running_stats=True)
    full_real_out = bn(data)
    loss = torch.sum(full_real_out[:batch_size // 2])
    loss.backward()
    real_grad = data.grad
    processes = []
    port = random.randint(25000, 30000)
    mini_batch_size = batch_size // size

    for rank in range(size):
        ctx = torch.multiprocessing.get_context("spawn")

        p = ctx.Process(target=init_process, args=(rank,
                                                   size, 
                                                   bn_test, 
                                                   data[rank * mini_batch_size: rank*mini_batch_size + mini_batch_size].detach(), 
                                                   full_real_out[rank * mini_batch_size: rank*mini_batch_size + mini_batch_size].detach(),
                                                   real_grad[rank * mini_batch_size: rank*mini_batch_size + mini_batch_size].detach(),
                                                   port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def bn_test(rank, input_chunk_, real_out, real_grad):
    input_chunk = input_chunk_.clone().detach()
    input_chunk.requires_grad = True
    sbn = SyncBatchNorm(input_chunk.shape[1])
    out = sbn(input_chunk)
    assert torch.allclose(out, real_out, atol=1e-3)
    if dist.get_world_size() == 1:
        loss = torch.sum(out[:out.shape[0] // 2])
    elif rank < dist.get_world_size() // 2:
        loss = torch.sum(out)
    else:
        loss = (0 * torch.sum(out))
    loss.backward()
    assert torch.allclose(input_chunk.grad, real_grad, atol=1e-1), rank


def init_process(rank, size, fn, input_chunk, real_out, real_grad, master_port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)

    fn(rank, input_chunk, real_out, real_grad)

