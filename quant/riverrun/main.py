from riverrun import RiverRun
import json


if __name__ == '__main__':

    config_path = 'json_file/build_sample.json'
    with open(config_path) as json_file:
        config = json.load(json_file)

    print('config = %s' % config)

    context = {
        "index_code": '000001.XSHG',
        "start_date": None,
        "end_date": None,
        "count": None,
        "sample_dt": 8,
        "label_dt_list": [1, 3, 7, 14, 28, 90],
        "feature_dt_list": [0, -1, -2, -3, -4, -5, -6, -7]
    }

    river_run = RiverRun(config_path, node_prefix='node.', local_mode=True)

    river_run.build_graph(context, [], {})









