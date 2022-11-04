# TensorRT2ONNX

A tool convert TensorRT engine/plan to a fake onnx

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

## Use [Netron](https://github.com/lutzroeder/netron) to view your fake onnx

![image](https://user-images.githubusercontent.com/92794867/199899590-4af79b85-2114-40f2-b43b-c8bcf71830e2.png)
