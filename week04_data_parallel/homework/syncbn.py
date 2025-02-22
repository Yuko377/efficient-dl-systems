import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        curr_sum = torch.sum(input, dim=0)
        curr_sum_of_sq = torch.sum(input ** 2, dim=0)
        stats = torch.cat([curr_sum, curr_sum_of_sq], dim=0) 
        
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        
        full_mean, full_mean_of_sq = (stats / (dist.get_world_size() * input.shape[0])).chunk(2)
        full_std = full_mean_of_sq - full_mean ** 2
        
        # if dist.get_rank() == 0:
        running_mean = (1 - momentum) * running_mean + momentum * full_mean
        running_std = (1 - momentum) * running_std + momentum * full_std
        
        ctx.save_for_backward(input, full_mean, full_std + eps)
        
        return (input - full_mean) / torch.sqrt(full_std + eps)
        

    @staticmethod
    def backward(ctx, grad_output):
        input, full_mean, full_std_eps = ctx.saved_tensors
        grad_normed_x = grad_output
        
        pre_grad_std = torch.sum(grad_normed_x * (input - full_mean), dim=0)  
        pre_grad_mu =  torch.sum(grad_normed_x, dim=0) 
        
        stats = torch.cat([pre_grad_std, pre_grad_mu], dim=0)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        pre_grad_std, pre_grad_mu = stats.chunk(2) 
        grad_std = -1/2 * full_std_eps ** (-3/2) * pre_grad_std
        grad_mu = -1 / torch.sqrt(full_std_eps) * pre_grad_mu        
        
        grad_inp = grad_normed_x / torch.sqrt(full_std_eps) + (grad_std * 2 * (input - full_mean) + grad_mu) / float(dist.get_world_size() * grad_output.shape[0])
        
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        
        return grad_inp, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        return sync_batch_norm.apply(input, self.running_mean, self.running_std, self.eps, self.momentum)
        
