"""
functions used to build dag
@Author Shen Zhouxu
@Email shenzhouxu@xiaohongshu.com
"""


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




