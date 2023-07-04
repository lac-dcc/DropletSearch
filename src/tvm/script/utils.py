import tvm
import numpy as np
from tvm import relay
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
import tvm.contrib.graph_executor as runtime

def get_network(name, batch_size, layout="NCHW", dtype="float32", use_sparse=False):

    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype, layout=layout
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=input_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=input_shape, num_classes=1000
        )
    elif name == 'bert':
        import torch, os
        import transformers  # pip3 install transfomers==3.0
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        model_class = transformers.BertModel
        tokenizer_class = transformers.BertTokenizer

        # You can also download them manualy
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
        # Then rename to pytorch_model.bin, vocab.txt & config.json
        # weight = 'path to downloaded model dir'
        weight = 'bert-base-uncased'
        model = model_class.from_pretrained(weight)
        model.eval()

        # tokenizer = tokenizer_class.from_pretrained(weight)
        # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
        # There is 30522 words in bert-base-uncased's vocabulary list
        input_shape = [batch_size, 128]
        input_name = 'input_ids'
        input_dtype = 'int64'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False)
        shape_list = [('input_ids', input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
                            tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse
        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)

    return mod, params, input_shape, output_shape

def get_best_time(log, ms=True):
    import json

    f = open(log, "r")
    best_avg, best_std, config = 9999.0, 0.0, ""

    for line in f.readlines():
        data = json.loads(line)
        r = np.mean(data["result"][0])
        if (best_avg > r):
            best_avg = r
            best_std = np.std(data["result"][0])
            config = data
    f.close()

    if ms: # convet to ms
        best_avg *= 1000
        best_std *= 1000
    return best_avg, best_std, config

def evaluate_performance(lib, data_shape, target, input_name="data", dtype="float32"):
    # upload parameters to device
    dev = tvm.device(str(target), 0)
    np.random.seed(0)
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype), device=dev)
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)
    # evaluate
    
    r = []
    for i in range(3):
        eval = module.benchmark(dev, number=5, repeat=5, min_repeat_ms=100, cooldown_interval_ms=100)
        r.append(eval.mean*1000)
    return r

def evaluate_p_value(lib, data_shape, target, input_name="data", dtype="float32"):
    dev = tvm.device(str(target), 0)
    np.random.seed(0)
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype), device=dev)
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)
    # evaluate
    r = []
    for i in range(3):
        eval = module.benchmark(dev, number=5, repeat=5, min_repeat_ms=100, cooldown_interval_ms=100)
        r.append(eval.mean*1000)
    return r