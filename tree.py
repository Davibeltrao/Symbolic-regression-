from anytree import Node, RenderTree
from random import randint

terminals = [3, 5]
functions = ['sum', 'sub', 'div', 'mul']
pop_list = []

class Tree(Node):
	num_childs = 0
	arity = 0 # 0 = Terminal / 1 = (exp, cos, sin) / 2 = (mul, div, sub, sum)
	def __init__(self, name):
		super().__init__(name)
		
	def isMaxDepth(self):
		if(self.depth >= 6):
			return True
		return False


def get_node_type(terminal_only=False):
	node_type = randint(0, 3)
	if node_type == 0 or terminal_only==True:
		terminal_node = randint(0, len(terminals)-1)
		return terminals[terminal_node]
	else:
		function_node = randint(0, len(functions)-1)
		return functions[function_node]	


def create_initial_pop(max_depth=2, node=None):
	if node == None:
		node_type = get_node_type()
		print(node_type)

		#create root node
		root = Tree(node_type)

		#get arity for root oeprator
		if node_type in terminals:
			root.arity = 0
		elif node_type in ['sin', 'cos', 'exp']:
			root.arity = 1
			#criar um filho recursivamente
			create_initial_pop(node=root)
		else:
			root.arity = 2
			#criar dois filhos recursivamente
			create_initial_pop(node=root)
			create_initial_pop(node=root)
		
		pop_list.append(root)
		return root
	
	elif node.depth == max_depth or node.arity == 0:
		print("Depth: ", node.depth, " Arity: ", node.arity)
		
		return

	elif node.depth == max_depth-1:
		node_type = get_node_type(terminal_only=True)
		if node.num_childs < node.arity:
			leaf = Tree(node_type)
			leaf.parent = node
			node.num_childs += 1
		return
	
	elif node.num_childs < node.arity:
		node.num_childs += 1
		node_type = get_node_type()
		leaf = Tree(node_type)
		leaf.parent = node

		#get arity for leaf operator
		if node_type in terminals:
			leaf.arity = 0
		elif node_type in ['sin', 'cos', 'exp']:
			leaf.arity = 1
			#criar somente um filho(arity 1)
			create_initial_pop(node=leaf)
		else:
			leaf.arity = 2
			#chamar recursivamente para criar dois nós filhos(arity 2)
			create_initial_pop(node=leaf)
			create_initial_pop(node=leaf)

		print(leaf)
		return


def calculate_tree_value(tree):
	if(tree.name == 'sub'):
		print("Sub Check")
		return calculate_tree_value(tree.children[0]) - calculate_tree_value(tree.children[1])
	if(tree.name == 'sum'):
		print("Sum Check")
		return calculate_tree_value(tree.children[0]) + calculate_tree_value(tree.children[1])
	if(tree.name == 'mul'):
		print("Mul Check")
		return calculate_tree_value(tree.children[0]) * calculate_tree_value(tree.children[1])
	if(tree.name == 'div'):
		print("Div Check")
		dividendo = calculate_tree_value(tree.children[1])
		print("Dividendo value: ", dividendo)
		if(dividendo >= -1 and dividendo <= 1):
			return 1
		return calculate_tree_value(tree.children[0]) / calculate_tree_value(tree.children[1])
	if(tree.name == 'x0'):
		#retornar valor de x0
		pass
	if(tree.name == 'x1'):
		#retornar valor de x1
		pass
	if(tree.name in terminals):
		print("Terminal Check")
		return int(tree.name)
	pass

tree = create_initial_pop()

for pre, fill, node in RenderTree(tree):
	print(pre, node.name)
	#pass

print(calculate_tree_value(tree))




