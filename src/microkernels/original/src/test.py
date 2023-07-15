import tvm, time
from tvm import relay
import numpy as np

import tvm.contrib.graph_executor as runtime

def matmul(shape):
    # Define the shape of the input tensors
    input_shape = shape

    # Create the input placeholders
    data1 = relay.var("data1", shape=input_shape, dtype="float32")
    data2 = relay.var("data2", shape=input_shape, dtype="float32")

    # Define the batch matrix multiplication operation
    mm = relay.nn.batch_matmul(data1, data2)

    # Create a Relay function with the defined inputs and operation
    func = relay.Function([data1, data2], mm)

    # Compile the function into a TVM module
    mod = tvm.IRModule.from_expr(func)

    # Set the target to be the default CPU
    target = tvm.target.Target("cuda")
    dev = tvm.cuda()

    # Build the TVM module
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target)

    # Create a TVM runtime module
    rt = runtime.GraphModule(lib["default"](dev))

    # Create random input data
    data1_np = np.random.uniform(size=input_shape).astype("float32")
    data2_np = np.random.uniform(size=input_shape).astype("float32")

    # Set the input data in the runtime module
    rt.set_input("data1", data1_np)
    rt.set_input("data2", data2_np)
    
    # Warmup and measure execution time
    rt.run()
    start_time = time.time()
    rt.run()
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000

    # Get the output tensor
    output = rt.get_output(0)

    # Print the output tensor
    #print(output)
    print(execution_time)

matmul((16,1024,1024))