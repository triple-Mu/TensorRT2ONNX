import json
import logging
import os
from collections import defaultdict
from typing import Dict, Union

import onnx
import tensorrt as trt

from .core import Json_Graph, TensorRT2ONNX

logger = logging.getLogger(__name__)


def build_onnx(reader: Union[str, Dict, trt.ICudaEngine],
               display_forking_regions: bool = False) -> onnx.ModelProto:
    if isinstance(reader, str):
        suffix = os.path.splitext(reader)[-1]
        if suffix in ('.engine', '.plan'):
            logger.warning('Processing a engine file into Json_Graph')
            trt_logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(trt_logger, '')
            with open(reader, 'rb') as f, trt.Runtime(trt_logger) as runtime:
                reader = runtime.deserialize_cuda_engine(f.read())
        elif suffix == '.json':
            logger.warning('Processing a json file into Json_Graph')
        else:
            raise TypeError(f'We do not support this Type:\n{suffix[1:]}')

    elif isinstance(reader, Dict):
        logger.warning('Processing a dict object into Json_Graph')
    elif isinstance(reader, trt.ICudaEngine):
        logger.warning('Processing an ICudaEngine object into Json_Graph')
    else:
        raise TypeError(f'We do not support this Type:\n{type(reader)}')

    if isinstance(reader, trt.ICudaEngine):
        read_layers = defaultdict(list)
        logger.warning('Processing a tensorrt engine into TRT_Graph ')
        Major, Branch, *SUF = map(int, trt.__version__.split('.'))
        assert Major == 8 and Branch >= 2, \
            f'Your tensorrt version: {trt.__version__} is mismatch'
        try:
            inspector = reader.create_engine_inspector()
        except Exception as e:
            logger.exception('There is a fault inside the tensorrt')
            logger.error(f'Error message is\n{e}')
        else:
            for i in range(reader.num_bindings):
                name = reader.get_binding_name(i)
                read_layers['Bindings'].append(name)
            for i in range(reader.num_layers):
                layer = inspector.get_layer_information(
                    i, trt.LayerInformationFormat.JSON)
                read_layers['Layers'].append(json.loads(layer))
        finally:
            del inspector, reader
            reader = read_layers.copy()

    json_graph = Json_Graph(reader)
    onnx_producer = TensorRT2ONNX(
        json_graph, display_forking_regions=display_forking_regions)
    return onnx_producer.onnx
