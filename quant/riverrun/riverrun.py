"""
build DAG in accordance with a config json file
@Author Jianfei Shen 
@Email sweetfishhcl@sina.com
"""

import tensorflow as tf
import importlib
import json
from graph.dag import collect_dependency_map, get_priority_map, invert_map


class RiverRun:
    def __init__(self, features, config_path, node_prefix='node.', local_mode=True):
        self.context = {}
        self.result_map = {}
        self.features = features
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
        node_fn = node_module.build_node_fn(params, self.features)
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

            self.info("finish building node: %s" % (prefix))
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

            self.result_map[prefix] = output
            self.info("finish building graph: %s" % (prefix))
        return output


if __name__ == '__main__':
    # unit test
    config_path = 'json_file/search4.json'
    with open(config_path) as json_file:
        config = json.load(json_file)

    print('config = %s' % config)
    param = {
        "task_types": ['click', 'like'],
        "is_train": True,
        "is_infer": False,
        "scope_dict":    {
            'op_dep_list': [],
            'dnn_scope_list': [],
            'seq_scope_list': [],
            'linear_scope_list': []
        },
        "node_prefix": 'node.',
        "logit_head_name_format": "head_%s",
        "only_dense": False,
        "dist": None
    }

    features = {'_dense_input_norm': tf.zeros([1280, 1874])}

    rank_model = RankModel(features, config_path, node_prefix='node.', local_mode=True)
    rank_model.context.update(param)
    rank_model.build_graph(prefix='Lagrange', config=config, input_list=[])

    multi_logits = rank_model.context['multi_logits']
    heads = rank_model.context['heads']

    head = tf.contrib.estimator.multi_head(heads)

    # spec = head.create_estimator_spec(
    #     features=features,
    #     mode=mode,
    #     labels=labels,
    #     logits=multi_logits,
    #     train_op_fn=_train_op_fn)


