import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        self.o1 = self.fc1 (x)
        self.o2 = self.relu1 (self.o1)
        o = self.fc2 (self.o2)
        return o


forward_values = {}
backward_values = {}

# Define the forward hook function
def hook_fn_forward(module, inp, out):
    forward_values[module] = {}
    forward_values[module]["input"] = inp
    forward_values[module]["output"] = out
    
def hook_fn_backward(module, inp_grad, out_grad):
    backward_values[module] = {}
    backward_values[module]["input"] = inp_grad
    backward_values[module]["output"] = out_grad

model = Model()

modules = model.named_children()
for name, module in modules:
    module.register_forward_hook(hook_fn_forward)
    module.register_backward_hook(hook_fn_backward)

# batch size of 1 -> shape = (1,3)
x = torch.tensor([[1.0, 1.0, 1.0]])
o = model(x)
o.backward()
print('yes')
