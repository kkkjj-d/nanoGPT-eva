import torch
# import horovod.torch as hvd
import torch.distributed as dist
import os
import enum


"""
Collective Communication Backend

Usage:
    import kfac.backend as backend
    
    hvd.init() or dist.init()
    backend.init()
    backend.comm.APIs()
"""


# global comm object
comm = None

# communicate operations
class Ops(enum.Enum):
    Average = "average"
    Sum = "sum"

# init backend
def init(backend):
    global comm
    if comm is None:
        comm = _get_comm_backend(backend)

def _get_comm_backend(backend):
        if backend == "Horovod":
            return RuntimeError('Horovod not surpported.')
            try:
                hvd.size()
                return _HorovodBackend()
            except:
                return RuntimeError('Horovod much be init before create HorovodBackend.')
        elif backend == "Torch":
            try:
                dist.get_world_size()
                return _TorchBackend()
            except:
                return RuntimeError('Torch.distributed much be init before create TorchBackend.')
        else:
            return RuntimeError('The backend is not implemented. Now only Horovod and Torch are supported.')


# class _HorovodBackend:
#     """
#     Collective communication backend based on Horovod
#     """
#     def __init__(self):
#         self.Average = Ops.Average
#         self.Sum = Ops.Sum

#     def size(self):
#         return hvd.size()

#     def local_rank(self):
#         return hvd.local_rank()

#     def rank(self):
#         return hvd.rank()
    
#     def new_group(self, ranks): # support process_sets after v0.23.0
#         return hvd.add_process_set(ranks)

#     def _get_op(self, op):
#         if op == Ops.Average:
#             return hvd.Average
#         elif op == Ops.Sum:
#             return hvd.Sum
#         else:
#             raise ValueError('Unknown communication operation {}'.format(op))
    
#     def allreduce(self, tensor, name=None, op=Ops.Average):
#         self.allreduce_(tensor, name, op)

#     def allreduce_(self, tensor, name=None, op=Ops.Average):
#         op = self._get_op(op)
#         hvd.allreduce_(tensor, name=name, op=op) # in-place synchronous all-reduce

#     def allreduce_async_(self, tensor, name=None, op=Ops.Average):
#         op = self._get_op(op)
#         return hvd.allreduce_async_(tensor, name=name, op=op) # in-place asynchronous all-reduce

#     def broadcast(self, tensor, src, group=None, name=None):
#         self.broadcast_(tensor, src, group, name)
    
#     def broadcast_(self, tensor, src, group=None, name=None):
#         if group is None:
#             hvd.broadcast_(tensor, root_rank=src, name=name) # in-place synchronous broadcast
#         else:
#             hvd.broadcast_(tensor, root_rank=src, process_set=group, name=name)
    
#     def broadcast_async_(self, tensor, src, group=None, name=None): # in-place asynchronous broadcast
#         if group is None:
#             return hvd.broadcast_async_(tensor, root_rank=src, name=name)
#         else:
#             return hvd.broadcast_async_(tensor, root_rank=src, process_set=group, name=name)

#     def synchronize(self, handle):
#         return hvd.synchronize(handle)



class _TorchBackend:
    """
    Collective communication backend based on Pytorch DDP
    """
    def __init__(self):
        self.Average = Ops.Average
        self.Sum = Ops.Sum

    def size(self):
        return dist.get_world_size()

    def local_rank(self):
        try:
            return int(os.environ['LOCAL_RANK'])
        except:
            raise RuntimeError('LOCAL_RANK must be set in the environment when using torch.distributed')

    def rank(self):
        return dist.get_rank()

    def new_group(self, ranks):
        return dist.new_group(ranks)
        
    def allreduce(self, tensor, name=None, op=Ops.Average):
        self.allreduce_(tensor, name, op)

    def allreduce_(self, tensor, name=None, op=Ops.Average):
        dist.all_reduce(tensor, async_op=False)
        if op == Ops.Average:
            tensor.div_(self.size())

    def allreduce_async_(self, tensor, name=None, op=Ops.Average):
        handle = dist.all_reduce(tensor, async_op=True)
        if op == Ops.Sum:
            return handle
        else:
            return (handle, tensor) # wait to be averaged

    def broadcast(self, tensor, src, group=None, name=None):
        self.broadcast_(tensor, src, group, name)
    
    def broadcast_(self, tensor, src, group=None, name=None):
        dist.broadcast(tensor, src=src, group=group, async_op=False)
    
    def broadcast_async_(self, tensor, src, group=None, name=None):
        return dist.broadcast(tensor, src=src, group=group, async_op=True)

    def synchronize(self, handle):
        if isinstance(handle, tuple):
            h, tensor = handle
            h.wait()
            tensor.div_(self.size())
        else:
            handle.wait()
        return dist.barrier()
        # return hvd.synchronize(handle)

def broadcast_parameters(params, root_rank):
        """
        Broadcasts the parameters from root rank to all other processes.
        Typical usage is to broadcast the `model.state_dict()`,
        `model.named_parameters()`, or `model.parameters()`.

        Arguments:
            params: One of the following:
                - list of parameters to broadcast
                - dict of parameters to broadcast
            root_rank: The rank of the process from which parameters will be
                    broadcasted to all other processes.
        """
        if isinstance(params, dict):
            params = sorted(params.items())
        elif isinstance(params, list):
            # support both named_parameters() and regular parameters()
            params = [p if isinstance(p, tuple) else (None, p) for p in params]
        else:
            raise ValueError('invalid params of type: %s' % type(params))

        # Run asynchronous broadcasts.
        for name, p in params:
            if p is not None:
                dist.broadcast(p.view(-1), root_rank)

def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if p is not None and not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, _ in params:
        if key in callbacks:
            callbacks[key]()