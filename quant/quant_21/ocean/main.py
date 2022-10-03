from dag import * 


config_path = 'config/test2.json'


with open(config_path) as f:
    config = json.load(f)
    f.close()

print('config = %s' % config)
build_graph(prefix='main', config=config, context={}, input_list=[], result_map={})
