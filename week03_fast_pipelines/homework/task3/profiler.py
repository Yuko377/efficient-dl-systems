import json
import time
import torch
import os
from collections import defaultdict


class Profile:
    def __init__(self, model, name="model", schedule=None):
        self.model = model
        self.name_map = self._build_name_map(model, name)
        self.events = []
        self.timers = {}
        self.hooks = []
        
        ### TODO
    
    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = f"{full_name}: {module.__class__.__name__}"#module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        ### TODO
        self.timers[f'fwd: {self.name_map[module]}'] = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self.timers[f'fwd: {self.name_map[module]}'].record()


    def _forward_post_hook(self, module, inputs, outputs):
        ### TODO
        ender = torch.cuda.Event(enable_timing=True)
        ender.record()
        torch.cuda.synchronize()
        self.events.append(('fwd', self.name_map[module], self.timers[f'fwd: {self.name_map[module]}'].elapsed_time(ender)))

    def _backward_pre_hook(self, module, grad_output):
        ### TODO
        self.timers[f'bwd: {self.name_map[module]}'] = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self.timers[f'bwd: {self.name_map[module]}'].record()
        
    def _backward_post_hook(self, module, grad_input, grad_output):
        ### TODO
        ender = torch.cuda.Event(enable_timing=True)
        ender.record()
        torch.cuda.synchronize()
        self.events.append(('bwd', self.name_map[module], self.timers[f'bwd: {self.name_map[module]}'].elapsed_time(ender)))
        
    def __enter__(self):
        ### TODO
        for _, module in self.model.named_modules():
            self.hooks.append(module.register_forward_pre_hook(self._forward_pre_hook))
            self.hooks.append(module.register_forward_hook(self._forward_post_hook))
            self.hooks.append(module.register_full_backward_pre_hook(self._backward_pre_hook))
            self.hooks.append(module.register_full_backward_hook(self._backward_post_hook))
 
    def __exit__(self, type, value, traceback):
        ### TODO
        self.summary()
        for handle in self.hooks:
            handle.remove()

    def step(self):
        ### TODO
        raise NotImplementedError

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        ### TODO
        raise NotImplementedError
