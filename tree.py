from anytree import Node, RenderTree
from random import randint

terminals = ['x0', 'x1']
functions = ['sum', 'sub', 'div', 'mul', 'sin', 'cos', 'exp']
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


def create_initial_pop(max_depth=7, node=None):
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





tree = create_initial_pop()

for pre, fill, node in RenderTree(tree):
	print("%s%s" % (pre, node.name))
	#pass



node_list = []


root = Tree("sum")
node_list.append(root)
n1 = Node("div", parent=node_list[0])
n2 = Node("4", parent=node_list[0])
n3 = Node("sum", parent=n1, foo=1)
Node("8", parent=n1)
Node("1", parent=n3)
Node("7", parent=n3)

for pre, fill, node in RenderTree(node_list[0]):
	print("%s%s" % (pre, node.name))



mary = Node("Mary")
urs = Node("Urs", parent=mary)
chris = Node("Chris", parent=urs)
marta = Node("Marta", parent=chris)
jhon = Node("Djon")
jhon.parent = marta

for pre, fill, node in RenderTree(root):
	print("%s%s" % (pre, node.name))
	#pass


#urs.parent = None
#chris = None

#n3.parent = chris

for pre, fill, node in RenderTree(mary):
	#print("%s%s" % (pre, node.name))
	pass



