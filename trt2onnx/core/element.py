from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

import numpy as np

_regionFormat: Dict[str, str] = {
    'Four wide channel vectorized row major Int8 format':
    'Int8 NC/4HW4',
    'Four wide channel vectorized row major FP32 format':
    'FP32 NC/4HW4',
    'Thirty-two wide channel vectorized row major Int8 format':
    'Int8 NC/32HW32',
    'Thirty-two wide channel vectorized row major FP32 format':
    'FP32 NC/32HW32',
    'Thirty-two wide channel vectorized row major FP16 format':
    'FP16 NC/32HW32',
    'Thirty-two wide channel vectorized row major Int8 format with 3 spatial dimensions': # noqa E501
    'Int8 NC32DHW',
    'Thirty-two wide channel vectorized row major FP16 format with 3 spatial dimensions': # noqa E501
    'FP16 NC32DHW',
    'Sixteen wide channel vectorized row major FP16 format':
    'FP16 NC16HW',
    'Channel major FP16 format where channel % 4 == 0':
    'FP16 NHWC4',
    'Channel major FP32 format where channel % 4 == 0':
    'FP32 NHWC4',
    'Channel major Int8 format where channel % 4 == 0':
    'Int8 NHWC4',
    'Channel major FP16 format where channel % 8 == 0':
    'FP16 NHWC8',
    'Channel major FP16 format where channel % 16 == 0':
    'FP16 NHWC16',
    'Channel major FP16 format where channel == 4 and column stride % 32 == 0':
    'FP16 NHWC4',
    'Channel major INT8 format where channel == 4 and column stride % 32 == 0':
    'Int8 NHWC4',
    'Channel major INT8 format where column stride % 32 == 0':
    'Int8 NHWC1',
    'Row major INT8 format where column stride % 64 == 0':
    'Int8 NCHW',
    'Channel major FP16 format where channel % 8 == 0 with 3 spatial dimensions': # noqa E501
    'FP16 NDHWC8',
    'Channel major FP16 format where channel == 1 and column stride % 32 == 0':
    'FP16 NHWC1',
    'Row major FP16 format where column stride % 64 == 0':
    'FP16',
    'Two wide channel vectorized row major FP16 format':
    'FP16 NC/2HW2',
    'Row major linear FP32':
    'FP32 NCHW',
    'Row major linear Int32':
    'INT32 NCHW',
    'Row major linear FP16 format':
    'FP16 NCHW',
    'Row major Int8 format':
    'Int8 NCHW',
    'Channel major FP32 format':
    'FP32 NHWC',
    'Channel major FP16 format':
    'FP16 NHWC',
    'Channel major Int8 format':
    'Int8 NHWC',
    'Row major linear BOOL':
    'Bool',
    'Unknown format':
    'Unknown format'
}

_type2Size: Dict[str, int] = {
    'Int8': 1,
    'Int32': 4,
    'Half': 2,
    'FP16': 2,
    'Float': 4,
    'FP32': 4,
    'INT32': 4,
    'Bool': 4,
    'Unknown format': 0
}


class Activation:

    def __init__(self, raw_dict: Dict):
        self.name = raw_dict['Name']
        self.shape = raw_dict['Dimensions']
        format = raw_dict['Format/Datatype']
        format = format.replace('.', '')
        self.format = _regionFormat.get(format, 'Unknown format')
        self.precision, self.data_size = self.parse_tensor_info()
        self.size_bytes = np.prod(self.shape) * self.data_size

    def parse_tensor_info(self):
        for _type, _size in _type2Size.items():
            if _type in self.format:
                return _type.upper(), _size
        raise ValueError(f'Uknown precision {self.format}')


class RegionMemOp(Enum):
    WRITE = 1
    READ = 2


class RegionEvent(NamedTuple):
    owner_layer: str
    port: int
    mem_op: RegionMemOp
    tensor: Activation


class PortDesc(NamedTuple):
    layer_name: Optional[str]
    port: Optional[int]


@dataclass
class RegionGeneration:
    tensor: Activation
    id: int
    is_user: bool = False
    is_forked: bool = False
    writers: List[PortDesc] = field(default_factory=list)
    readers: List[PortDesc] = field(default_factory=list)


class Region:

    def __init__(self):
        self.__generations: List[RegionGeneration] = list()
        self.name: str = ''

    @property
    def generations(self):
        return self.__generations

    def add_generation(self, tensor: Activation):
        self.name = self.name or tensor.name
        gen_id = len(self.generations)
        self.__generations.append(RegionGeneration(tensor, gen_id))

    def nb_generations(self):
        return len(self.__generations)

    def __update_shape(self, tensor: Activation, generation: int):
        if tensor.size_bytes > \
                self.__generations[generation].tensor.size_bytes:
            self.__generations[generation].tensor = tensor

    def add_writer(self, gen_id: int, writer: PortDesc, tensor: Activation):
        self.__generations[gen_id].writers.append(writer)
        self.__update_shape(tensor, gen_id)

    def add_reader(self, gen_id: int, reader: PortDesc, tensor: Activation):
        self.__generations[gen_id].readers.append(reader)
        self.__update_shape(tensor, gen_id)
        if gen_id > 0:
            self.__update_shape(tensor, gen_id - 1)

    def writers(self, gen_id: int = None):
        try:
            if gen_id is not None:
                return self.__generations[gen_id].writers
            # Return a list of writers from all generations
            writers = []
            for generation in self.__generations:
                writers.extend(generation.writers)
            return writers
        except KeyError:
            return []

    def readers(self, gen_id: Optional[int]):
        try:
            if gen_id is not None:
                return self.__generations[gen_id].readers
            # Return a list of readers from all generations
            readers = []
            for generation in self.__generations:
                readers.extend(generation.readers)
            return readers
        except KeyError:
            return []

    def is_placeholder(self) -> bool:
        return len(self.generations) > 1


class Edge(NamedTuple):
    src: PortDesc
    dst: PortDesc
    tensor: Activation
    region_gen: int


class MemoryNode(NamedTuple):
    name: str
    tensor: Activation
    region_gen: int
    is_user: bool


class Layer:

    def __init__(self, raw_dict: Dict):
        self.raw_dict = raw_dict
        self.name = raw_dict['Name']
        self.type = raw_dict.get('ParameterType') or raw_dict.get('LayerType')
        self.subtype = raw_dict['LayerType']
        self.inputs = [Activation(tensor) for tensor in raw_dict['Inputs']]
        self.outputs = [Activation(tensor) for tensor in raw_dict['Outputs']]
        self.outputs_size_bytes = np.sum([i.size_bytes for i in self.outputs])
        if self.inputs:
            self.precision = self.inputs[0].precision
            self.inputs_size_bytes = np.sum(
                [i.size_bytes for i in self.inputs])
        else:
            self.inputs_size_bytes = 0
            self.precision = None

        self.total_io_size_bytes = \
            self.inputs_size_bytes + self.outputs_size_bytes
        self._parse_weights()
        self.total_footprint_bytes = \
            self.total_io_size_bytes + self.weights_size

    @staticmethod
    def _parse_constant(weights):
        if weights is None:
            cnt, data_type = 0, None
        else:
            cnt = weights['Count']
            data_type = weights['Type']
        data_size = _type2Size.get(data_type, 0)
        return cnt, data_type, cnt * data_size

    def _parse_weights(self):
        weights = self.raw_dict.get('Weights')
        bias = self.raw_dict.get('Bias')
        self.weights_cnt, self.weights_type, self.weights_size = \
            self._parse_constant(weights)
        self.bias_cnt, self.bias_type, self.bias_size = \
            self._parse_constant(bias)


class LayerNode(NamedTuple):
    layer: Layer
