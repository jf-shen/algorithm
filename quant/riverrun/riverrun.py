"""
build DAG in accordance with a config json file
@Author Jianfei Shen 
@Email sweetfishhcl@sina.com
"""

import importlib
import json
from dag import collect_dependency_map, get_priority_map, invert_map


class RiverRun:
    def __init__(self, config_path, node_prefix='node.', local_mode=True):
        self.context = {}
        self.result_map = {}
        self.node_prefix = node_prefix

        self.local_mode = local_mode
        self.graph_name = 'RiverRun'




        with open(config_path) as f:
            self.config = json.load(f)
            f.close()

    def info(self, info_str):
        info_str = '[%s] %s' % (self.graph_name, info_str) 
        if self.local_mode:
            print(info_str)
        else:
            raise NotImplementedError("only support local mode, info: %s" % info_str)

    def build_node_fn(self, node_type, params):
        node_module = importlib.import_module(self.node_prefix + node_type)
        node_fn = node_module.build_node_fn(params)
        return node_fn

    def build_graph(self, prefix, config, input_list):
        """
        build graph/subgraph/node
        Args:
            prefix: a prefix to specify graphs/subgraphs/nodes
            config: a json instruction on how to build the graph
            input_list: a list of input tensor
        Returns:
            output: a tensor
        """

        assert type(input_list) == list, 'type(input_list) = %s, prefix = %s' % (type(input_list), prefix)

        if config.get("dag") is None:
            node_type = config.get("type")
            params = {"prefix": prefix}
            params.update(config)

            self.info("building node: %s, params = %s" % (prefix, params))

            node_fn = self.build_node_fn(node_type, params)
            output = node_fn(self.context, input_list, self.result_map)

            self.info("finish building node: %s" % prefix)
            if config.get("save_result"):
                self.result_map[prefix] = output
        else:
            # build sub-graph
            self.info("start building graph: %s" % prefix)

            dag = config.get("dag")
            nodes = config.get("nodes")

            assert nodes is not None, config

            dependency_map = collect_dependency_map(dag)
            priority_map = get_priority_map(dependency_map)

            level_map = invert_map(priority_map)
            max_level = max(level_map.keys())

            local_result_map = dict()

            for i in range(max_level):
                level = i + 1
                level_node_list = level_map[level]
                for node_name in level_node_list:
                    input_name_list = dependency_map[node_name]
                    input_list = [local_result_map[input_name] for input_name in input_name_list]
                    output = self.build_graph(prefix=prefix + '/' + node_name,
                                              config=nodes.get(node_name),
                                              input_list=input_list)
                    local_result_map[node_name] = output

            if config.get('save_result'):
                self.result_map[prefix] = output
            self.info("finish building graph: %s" % prefix)
        return output


if __name__ == '__main__':
    # unit test
    config_path = 'json_file/search4.json'
    with open(config_path) as json_file:
        config = json.load(json_file)

    print('config = %s' % config)


