import json
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from .element import (Activation, Edge, Layer, LayerNode, MemoryNode, PortDesc,
                      Region, RegionEvent, RegionGeneration, RegionMemOp)


class Json_Graph:

    def __init__(self, graph_info: Union[str, Dict]):
        if isinstance(graph_info, str):
            with open(graph_info) as f:
                graph = json.load(f)
        elif isinstance(graph_info, Dict):
            graph = graph_info.copy()
        else:
            raise TypeError

        self.bindings = graph['Bindings']
        self.raw_layers = graph['Layers']
        self.convert_deconv()
        self.rename_layer_with_count()
        self.create_layers()

    def convert_deconv(self):
        for raw_layer in self.raw_layers:
            isconv = raw_layer.get('ParameterType') == 'Convolution'
            isdeconv = raw_layer.get('LayerType') == 'CaskDeconvolutionV2'
            if isconv and isdeconv:
                raw_layer['ParameterType'] = 'Deconvolution'

    def rename_layer_with_count(self):
        names_cnt = defaultdict(int)
        for raw_layer in self.raw_layers:
            name = raw_layer['Name']
            if names_cnt[name]:
                name += '_' + str(names_cnt[name])
            raw_layer['Name'] = name

    def create_layers(self):
        layers = [Layer(raw_layer) for raw_layer in self.raw_layers]
        self.fold_no_ops(layers)
        self.all_layers = self.layers.copy()
        constants = []
        no_constant_layers = []
        for layer in layers:
            if layer.type == 'Constant':
                constants.append(layer)
            else:
                no_constant_layers.append(layer)
        self.constants, self.layers = constants, no_constant_layers

    @staticmethod
    def consumers_producers_dict(layers):
        consumers, producers = defaultdict(list), defaultdict(list)
        for layer in layers.values():
            for i, inp in enumerate(layer.inputs):
                consumers[inp.name].append((layer.name, i))
            for i, out in enumerate(layer.outputs):
                producers[out.name].append((layer.name, i))

        return consumers, producers

    @staticmethod
    def move_input(src: Layer, dst: Layer, index: int = 0):
        dst.inputs[index] = src.inputs[0]

    @staticmethod
    def move_output(src: Layer, dst: Layer, index: int = 0):
        dst.outputs[index] = src.outputs[0]

    def fold(self, no_op: Layer):
        consumers = self.activation_consumers.get(no_op.outputs[0].name)
        is_output = no_op.outputs[0].name in self.bindings
        if consumers is not None:
            for consumer in consumers:
                name, index = consumer
                self.move_input(src=no_op,
                                dst=self.name2layer[name],
                                index=index)
        else:
            if is_output:
                outputs = self.activation_producers.get(no_op.inputs[0].name)
                for output in outputs:
                    name, index = output
                    self.move_output(src=no_op,
                                     dst=self.name2layer[name],
                                     index=index)

    def fold_no_ops(self, layers: List[Layer]):
        name2layer = {layer.name: layer for layer in layers}
        self.activation_consumers, self.activation_producers = \
            self.consumers_producers_dict(name2layer)
        self.name2layer = name2layer
        for layer in layers:
            if layer.type == 'NoOp':
                self.fold(layer)
        self.layers = [
            layer for layer in self.name2layer.values() if layer.type != 'NoOp'
        ]

    def get_bindings(self) -> Tuple[List[Activation], List[Activation]]:
        inputs, outputs = [], []
        seen = set()
        for layer in self.layers:
            for inp in layer.inputs:
                if inp.name in self.bindings and inp.name not in seen:
                    inputs.append(inp)
                    seen.add(inp.name)
            for out in layer.outputs:
                if out.name in self.bindings and out.name not in seen:
                    outputs.append(out)
                    seen.add(out.name)
        return inputs, outputs


class TensorRT_Graph:

    def __init__(
        self,
        json_graph: Json_Graph,
        include_forking_regions: bool = False,
    ):
        self.include_forking_regions = include_forking_regions
        self.json_graph = json_graph
        self.regions = self.regions_factory()
        self._edges_list: List[Edge] = []
        self._layers_nodes: List[Layer] = []
        self._memory_nodes: List[MemoryNode] = []
        self.__create_graph()

    @property
    def layers_nodes(self):
        return self._layers_nodes

    @property
    def memory_nodes(self):
        return self._memory_nodes

    @property
    def edges_list(self):
        return self._edges_list

    def set_is_user(self, region_name, region: Region):
        n_generations = len(region.generations)
        for gen_id, generation in enumerate(region.generations):
            nb_writers = len(generation.writers)
            nb_readers = len(generation.readers)
            is_binding = region_name in self.json_graph.bindings
            if gen_id == n_generations - 1:
                generation.is_user = is_binding
            else:
                generation.is_user = is_binding and (nb_readers == 0
                                                     or nb_writers == 0)

    @staticmethod
    def set_is_forked(region: Region):
        for generation in region.generations:
            nb_writers = len(generation.writers)
            nb_readers = len(generation.readers)
            generation.is_forked = nb_writers > 1 or nb_readers > 1

    def regions_factory(self):
        story = defaultdict(list)

        for layer in self.json_graph.layers:
            for i, inp in enumerate(layer.inputs):
                region_event = RegionEvent(layer.name, i, RegionMemOp.READ,
                                           inp)
                story[inp.name].append(region_event)
            for i, out in enumerate(layer.outputs):
                region_event = RegionEvent(layer.name, i, RegionMemOp.WRITE,
                                           out)
                story[out.name].append(region_event)

        regions = []
        for region_name, region_evts in story.items():
            region = Region()
            current_gen = -1
            previous_mem_op = None
            for evt in region_evts:
                if evt.mem_op != previous_mem_op:
                    if evt.mem_op == RegionMemOp.WRITE or not previous_mem_op:
                        current_gen += 1
                        region.add_generation(evt.tensor)
                evt_layer = PortDesc(evt.owner_layer, evt.port)
                if evt.mem_op == RegionMemOp.WRITE:
                    region.add_writer(current_gen, evt_layer, evt.tensor)
                else:
                    region.add_reader(current_gen, evt_layer, evt.tensor)
                previous_mem_op = evt.mem_op

            self.set_is_user(region_name, region)
            self.set_is_forked(region)
            regions.append(region)
        return regions

    def should_include_region(self, region: Region,
                              generation: RegionGeneration,
                              is_constant: bool) -> bool:
        nb_gens = region.nb_generations()
        is_user = generation.is_user
        is_forked = self.include_forking_regions and generation.is_forked
        include = is_user or nb_gens > 1 or is_forked
        return not is_constant and include

    @staticmethod
    def make_memory_node_name(region: Region,
                              generation: RegionGeneration) -> str:
        if generation.is_user:
            return region.name
        return '.'.join((region.name, str(generation.id)))

    def __connect_writer_to_all_readers(self, writer_desc: PortDesc,
                                        generation: RegionGeneration):
        for reader in generation.readers:
            self._edges_list.append(
                Edge(writer_desc, PortDesc(reader.layer_name, reader.port),
                     generation.tensor, generation.id))

    def __add_region_bypass_edges(self, generation: RegionGeneration):
        for writer in generation.writers:
            writer_port = None if generation.is_user else writer.port
            writer_desc = PortDesc(writer.layer_name, writer_port)
            self.__connect_writer_to_all_readers(writer_desc, generation)

    def __add_layer_nodes(self):
        self._layers_nodes = [
            LayerNode(layer) for layer in self.json_graph.all_layers
        ]

    def __add_constant_node(self, region: Region, generation: RegionGeneration,
                            constants_producers):
        assert not generation.writers
        activation_name = self.make_memory_node_name(region, generation)
        constant = constants_producers[activation_name]
        self.__connect_writer_to_all_readers(PortDesc(constant.name, 0),
                                             generation)

    def __add_inter_region_edges(self):
        for region in self.regions:
            if len(region.generations) == 1:
                continue
            prev_generation = region.generations[0]
            for gen_id in range(1, len(region.generations)):
                curr_generation = region.generations[gen_id]
                self._edges_list.append(
                    Edge(
                        PortDesc(
                            self.make_memory_node_name(region,
                                                       prev_generation), 0),
                        PortDesc(
                            self.make_memory_node_name(region,
                                                       curr_generation), 0),
                        curr_generation.tensor, gen_id))
                prev_generation = curr_generation

    def __add_ingress_edges(self, region: Region,
                            generation: RegionGeneration):
        node_name = self.make_memory_node_name(region, generation)
        node_port = None if generation.is_user else 0
        for writer in generation.writers:
            self._edges_list.append(
                Edge(PortDesc(writer.layer_name, writer.port),
                     PortDesc(node_name, node_port), generation.tensor,
                     generation.id))

    def __add_egress_edges(self, region: Region, generation: RegionGeneration):
        activation_name = self.make_memory_node_name(region, generation)
        activation_port = None if generation.is_user else 0
        self.__connect_writer_to_all_readers(
            PortDesc(activation_name, activation_port), generation)

    def __add_memory_node(self, region, generation):
        is_user = generation.is_user
        node_name = self.make_memory_node_name(region, generation)
        self._memory_nodes.append(
            MemoryNode(node_name, generation.tensor, generation.id, is_user))
        self.__add_ingress_edges(region, generation)
        self.__add_egress_edges(region, generation)

    def __add_memory_nodes(self):
        constants_outputs = [
            const.outputs[0].name for const in self.json_graph.constants
        ]
        constants_producers = {
            const.outputs[0].name + '.0': const
            for const in self.json_graph.constants
        }
        for region in self.regions:
            is_constant = region.name in constants_outputs
            if is_constant:
                continue
            for generation in region.generations:
                include_region = self.should_include_region(
                    region, generation, is_constant)
                if not include_region:
                    self.__add_region_bypass_edges(generation)
                    continue
                if is_constant:
                    self.__add_constant_node(region, generation,
                                             constants_producers)
                else:
                    self.__add_memory_node(region, generation)

    def __create_graph(self):
        self.__add_layer_nodes()
        self.__add_memory_nodes()
        self.__add_inter_region_edges()
