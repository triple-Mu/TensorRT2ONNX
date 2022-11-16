from collections import defaultdict
from typing import Dict, List, Tuple

import onnx
from onnx.helper import (make_attribute, make_graph, make_model, make_node,
                         make_opsetid, make_tensor_value_info)

from .graph import Json_Graph, Layer, TensorRT_Graph


class TensorRT2ONNX:

    def __init__(self, json_graph: Json_Graph, display_forking_regions=False):
        self.onnx_nodes: List = []
        self.graph_inputs: List = []
        self.graph_outputs: List = []
        self.json_graph = json_graph
        self.trt_graph = TensorRT_Graph(
            json_graph, include_forking_regions=display_forking_regions)
        self.inputs_map, self.outputs_map = self.init_list()
        self.add_memory_nodes()
        self.add_layer_nodes()
        self.add_graph_inputs_outputs()
        self.__onnx_model = self.finalize_onnx_graph()

    def init_list(self) -> Tuple[Dict[str, List], Dict[str, List]]:
        inputs_map, outputs_map = defaultdict(list), defaultdict(list)
        for edge in self.trt_graph.edges_list:
            if edge.src.port is not None and \
                    edge.dst.port is not None:
                edge_name = f'{edge.src.layer_name}:{edge.src.port}' \
                            f'\t##\t{edge.dst.layer_name}:{edge.dst.port}'
            elif edge.src.port is not None:
                edge_name = edge.dst.layer_name
            else:
                edge_name = edge.src.layer_name

            outputs_map[edge.src.layer_name].append(edge_name)
            inputs_map[edge.dst.layer_name].append(edge_name)
        return inputs_map, outputs_map

    def finalize_onnx_graph(self) -> onnx.ModelProto:
        graph_def = make_graph(self.onnx_nodes, 'trt-onnx-graph',
                               self.graph_inputs, self.graph_outputs)
        return make_model(graph_def,
                          producer_name='triplemu',
                          opset_imports=[make_opsetid(domain='', version=20)])

    def __add_region_node(self, gen: int, region_name: str, is_user: bool,
                          inputs, outputs):
        assert isinstance(gen, int)
        assert not is_user
        node_def = make_node('Region', inputs, outputs, region_name)
        self.onnx_nodes.append(node_def)

    def add_memory_nodes(self):
        for mem_node in self.trt_graph.memory_nodes:
            is_user = mem_node.is_user
            if not is_user:
                self.__add_region_node(mem_node.region_gen, mem_node.name,
                                       is_user, self.inputs_map[mem_node.name],
                                       self.outputs_map[mem_node.name])

    @staticmethod
    def get_op_type(layer):
        op_type = layer.type
        if op_type == 'Convolution':
            op_type = 'Conv'
        if op_type == 'Pooling':
            if layer.raw_dict['PoolingType'] == 'AVERAGE':
                op_type = 'AveragePool'
            if layer.raw_dict['PoolingType'] == 'Max':
                op_type = 'MaxPool'
        return op_type

    @staticmethod
    def get_tensor_type(desc):
        desc = desc.lower()
        if 'int8' in desc:
            return onnx.TensorProto.INT8
        elif 'fp32' in desc:
            return onnx.TensorProto.FLOAT
        elif 'fp16' in desc:
            return onnx.TensorProto.FLOAT16
        elif 'int32' in desc:
            return onnx.TensorProto.INT32
        else:
            print(f'Uknown precision {desc}')
            print('Set tensor type float default ')
            return onnx.TensorProto.FLOAT

    @staticmethod
    def add_attributes(layer, node_def):
        for key, value in sorted(layer.items()):
            if key not in [
                    'InputRegions', 'OutputRegions', 'Inputs', 'Outputs',
                    'Name', 'name', 'ParameterType', 'LayerName'
            ]:
                node_def.attribute.extend([make_attribute(key, value)])

    def make_onnx_tensor(self, tensor):
        t = make_tensor_value_info(tensor.name,
                                   self.get_tensor_type(tensor.format),
                                   tensor.shape)
        return t

    def __add_layer_node(self, node_id: int, layer: Layer, inputs, outputs):
        assert isinstance(node_id, int)
        op_type = self.get_op_type(layer)
        if op_type == 'Constant':
            return

        node_def = make_node(op_type, inputs, outputs, layer.name)
        self.add_attributes(layer.raw_dict, node_def)
        self.onnx_nodes.append(node_def)

    def add_layer_nodes(self):
        for layer_id, layer_node in enumerate(self.trt_graph.layers_nodes):
            layer = layer_node.layer
            try:
                self.__add_layer_node(layer_id, layer,
                                      self.inputs_map[layer.name],
                                      self.outputs_map[layer.name])
            except Exception:
                continue

    def add_graph_inputs_outputs(self):
        g_inputs, g_outputs = self.json_graph.get_bindings()
        for inp in g_inputs:
            self.graph_inputs.append(self.make_onnx_tensor(inp))
        for outp in g_outputs:
            self.graph_outputs.append(self.make_onnx_tensor(outp))

    @property
    def onnx(self):
        return self.__onnx_model
