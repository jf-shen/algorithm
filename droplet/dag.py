"""
build DAG in accordance with a config json file 
@Author Jianfei Shen
@Email sweetfishhcl@sina.com
"""

import json
import importlib


def build_node_fn(node_type, params):
    node_module = importlib.import_module('node.' + node_type)
    node_fn = node_module.build_node_fn(params)
    return node_fn


def collect_dependency_map(dag):
    """
    Args:
        dag: a json map of {father_node: child_node/[child_nodes]} 
    Return:
        dependency_map: {node: [father_nodes]}
    """
    dependency_map = dict() 
    for k, v_maybe_list in dag.items():
        if dependency_map.get(k) is None:
            dependency_map[k] = [] 

        v_list = v_maybe_list if type(v_maybe_list) == list else [v_maybe_list]
        for v in v_list:
            if dependency_map.get(v) is None:
                dependency_map[v] = [k]
            else:
                dependency_map[v].append(k)
    return dependency_map


def get_priority_map(dependency_map):
    """
    Args:
        dependency_map: a dict of {node: [father_nodes]}
    Returns:
        priority_map: a dict of node generation priority: {node: priority}, with priority started from 1
    """

    priority_map = dict() 
    remain = 0

    for k in dependency_map.keys():
        priority_map[k] = -1
        remain += 1

    while remain > 0:
        for k, v_list in dependency_map.items():
            if priority_map.get(k) >= 0:
                continue

            ready_flag = True
            max_priority = 0
            for v in v_list:
                if priority_map.get(v) < 0:
                    ready_flag = False
                    break
                else:
                    max_priority = max(max_priority, priority_map[v])

            if ready_flag:
                priority_map[k] = max_priority + 1 
                remain -= 1 
    return priority_map


def invert_map(input_map): 
    """
    invert {k:v} map and group by v
    Args:
        input_map: {k, v}
    Return:
        output_map: {v, [k]}
    """
    output_map = dict() 
    for k, v in input_map.items():
        if output_map.get(v) is None:
            output_map[v] = [k]
        else:
            output_map[v].append(k) 
    return output_map


def build_graph(prefix, config, context, input_list, result_map):
    """
    build graph / subgraph / node
    Args:
        prefix: a prefix to specify graphs / subgraphs/ nodes 
        config: a json-format dict to instruct how to build the graph 
        context:
        input_list: a list of input dataset 
        result_map: store all the previous results
    Returns:
        output dataSet
    """

    assert type(input_list) == list, 'type(input_list) = %s, prefix = %s' % (type(input_list), prefix)

    if config.get("dag") is None:
        print("building node: %s" % prefix)
        node_type = config.get("type")

        params = {"prefix": prefix}
        params.update(config)

        node_fn = build_node_fn(node_type, params)
        output = node_fn(context, input_list, result_map)

        result_map[prefix] = output
    else:
        # build sub-graph 
        print("start building graph: %s" % prefix)

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
                output = build_graph(prefix=prefix + '/' + node_name, 
                                     config=nodes.get(node_name), 
                                     context=context,
                                     input_list=input_list, 
                                     result_map=result_map)
                local_result_map[node_name] = output

        result_map[prefix] = output     
        print("finish building graph: %s" % prefix)
    

if __name__ == '__main__':

    config_path = 'config/test.json'

    with open(config_path) as f:
        config = json.load(f)
        f.close()

    print('config = %s' % config)
    build_graph(prefix='main', config=config, context={}, input_list=[], result_map={})

