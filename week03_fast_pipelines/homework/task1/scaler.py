import torch


class StaticScaler:
    def __init__(self, init_scale_factor):
        self.scale_factor = init_scale_factor
        
    def scale(self, loss_tensor):
        return self.scale_factor * loss_tensor
    
    def step(self, optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                p.grad *= 1 / self.scale_factor
        optimizer.step()            
                        
    def update(self):
        pass


class DynamicScaler:
    def __init__(self, init_scale_factor, up_multiplier, down_multiplier, threshold):
        self.scale_factor = init_scale_factor
        self.up_multiplier = up_multiplier
        self.down_multiplier = down_multiplier
        self.curr_streak = 0
        self.threshold = threshold
        
    def scale(self, loss_tensor):
        return self.scale_factor * loss_tensor
    
    def step(self, optimizer):
        found_invalid_grads = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if not torch.all(torch.isfinite(p.grad)):
                    found_invalid_grads = True
                    break
            if found_invalid_grads:
                break
                
        ### check dtype of grads?
        if not found_invalid_grads:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.grad *= 1 / self.scale_factor
            optimizer.step()
            self.curr_streak += 1
        else:
            self.curr_streak = 0
                                    
    def update(self):
        if self.curr_streak == 0:
            self.scale_factor *= self.down_multiplier

        if self.curr_streak == self.threshold:
            self.scale_factor *= self.up_multiplier
            self.curr_streak = 0
        