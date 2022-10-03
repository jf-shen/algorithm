

def build_node_fn(params):

	alias = params.get('alias', 'no alias ...')

	def node_fn(context, input_list, result_map):
		print('alias = %s' % alias)
		print('input_list = %s' % input_list)
		print('result_map = %s' % result_map)

		return None if len(input_list) == 0 else input_list[0]

	return node_fn


