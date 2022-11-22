# TensorRT2ONNX

A tool convert TensorRT engine/plan to a fake onnx

## Build an engine using C++ or Python api

Set building config with `DETAILED` flag.

### C++

```cpp
config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
```

### Python

```python
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
```

## Build an engine from onnx using trtexec tools

```shell
trtexec --verbose \
        --nvtxMode=verbose \
        --buildOnly \
        --workspace=8192 \
        --onnx=your_onnx.onnx \
        --saveEngine=your_engine.engine \
        --timingCacheFile=timing.cache \
        --fp16 # use fp16
```

Notice: `--nvtxMode=verbose` is the same as `--profilingVerbosity=detailed`

You will get a `your_engine.engine` and a `timing.cache`

## Parser network from engine using trtexec tools

```shell
trtexec --verbose \
        --noDataTransfers \
        --useCudaGraph \
        --separateProfileRun \
        --useSpinWait \
        --nvtxMode=verbose \
        --loadEngine=your_engine.engine \
        --exportLayerInfo=graph.json \
        --timingCacheFile=timing.cache
```

You will parser `your_engine.engine` network information into `graph.json`

## Install TensorRT2ONNX

```shell
pip3 install trt2onnx -i https://pypi.org/simple
```

## Build a fake onnx from graph json

```python
import onnx
from trt2onnx import build_onnx

# build a fake onnx from json
onnx_graph = build_onnx('graph.json')

# save the fake onnx as `fake.onnx`
onnx.save(onnx_graph, 'fake.onnx')
```

## Build a fake onnx from engine

You must build engine with flag `ProfilingVerbosity=DETAILED`.

```python
import onnx
from trt2onnx import build_onnx

# build a fake onnx from engine
onnx_graph = build_onnx('your_engine.engine')

# save the fake onnx as `fake.onnx`
onnx.save(onnx_graph, 'fake.onnx')
```

**NOTICE !!**

If you build engine use your own plugin,
please load the `*.so` before `build_onnx` function.

```python
import ctypes
# load your plugin first
ctypes.cdll.LoadLibrary('your_plugin_0.so')
ctypes.cdll.LoadLibrary('your_plugin_1.so')
...
```

## A demo for resnet50

```python
import torch
import onnx
from trt2onnx import build_onnx
import tensorrt as trt
from torchvision.models import resnet50, ResNet50_Weights
device = torch.device('cuda:0')
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet.eval()
fake_input = torch.randn(1,3,224,224).to(device)
# dry run
resnet(fake_input)
# export onnx you will get `resnet50.onnx`
torch.onnx.export(resnet, fake_input, 'resnet50.onnx', opset_version=11)
# build engine
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.max_workspace_size = torch.cuda.get_device_properties(device).total_memory
flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)
parser.parse_from_file('resnet50.onnx')
# fp16 export
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
# set detail flag
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
# get `resnet50.engine`
with open('resnet50.engine','wb') as f, builder.build_engine(network, config) as engine:
    f.write(engine.serialize())
# get fake onnx
fake_onnx = build_onnx('resnet50.engine')
# save fake onnx
onnx.save(fake_onnx, 'fake_onnx.onnx')
```

## Use [Netron](https://github.com/lutzroeder/netron) to view your fake onnx

![image](https://user-images.githubusercontent.com/92794867/199899590-4af79b85-2114-40f2-b43b-c8bcf71830e2.png)
