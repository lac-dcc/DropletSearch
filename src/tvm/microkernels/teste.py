import tvm
from tvm import relay
import torch
import numpy as np

# Define the PyTorch Conv1D model
class Conv1DModel(torch.nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

# Create an instance of the PyTorch model
torch_model = Conv1DModel()

# Set the model to evaluation mode
torch_model.eval()

# Generate a random input tensor
input_shape = (1, 1, 10)  # (batch_size, channels, sequence_length)
input_data = np.random.rand(*input_shape).astype(np.float32)

# Convert the PyTorch model to TVM
input_name = "input"
shape_dict = {input_name: input_data.shape}
mod, params = relay.frontend.from_pytorch(torch_model, shape_dict)

# Set the target device and build the TVM module
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

# Create a TVM runtime module
ctx = tvm.cpu(0)
module = tvm.contrib.graph_runtime.create(graph, lib, ctx)

# Set the input data
module.set_input(input_name, tvm.nd.array(input_data))

# Run the model
module.run()

# Get the output tensor
output = module.get_output(0)

# Convert the output tensor to a numpy array
output_data = output.asnumpy()

print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)