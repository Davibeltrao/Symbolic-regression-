from anytree import Node, RenderTree
from random import randint
import os
import csv
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


gen_size = 50
terminals = ['x0', 'x1']
functions = ['sum', 'sub', 'div', 'mul']

pop_list = []
pop_fitness = []

dataset_path = 'datasets/synth1/synth1-train.csv'


class FileLoad:
	X_train = []
	y_train = []
	actual_index = 0
		
	def __init__(self, file):
		data = np.genfromtxt(file, delimiter=',')
		self.X_train = data[:, :-1]
		print("X: ", self.X_train)
		self.y_train = data[:, -1]
		print("y: ", self.y_train)

	def getData(self):
		X_return  = self.X_train[self.actual_index]
		y_return = self.y_train[self.actual_index]
		
		self.actual_index += 1
		self.actual_index = self.actual_index % 60
		
		return self.X_train, self.y_train

class Tree(Node):
	num_childs = 0
	arity = 0 # 0 = Terminal / 1 = (exp, cos, sin) / 2 = (mul, div, sub, sum)
	def __init__(self, name):
		super().__init__(name)
				
	def isMaxDepth(self):
		if(self.depth >= 6):
			return True
		return False


	def __str__(self, level=0):
		ret = "    "*level+repr(self.name)+"\n"
		for child in self.children:
			ret += child.__str__(level+1)
		return ret

	def __repr__(self):
		return '<tree node representation>'


def get_node_type(terminal_only=False):
	node_type = randint(0, 5)
	if node_type == 0 or terminal_only==True:
		terminal_node = randint(0, len(terminals)-1)
		return terminals[terminal_node]
	else:
		function_node = randint(0, len(functions)-1)
		return functions[function_node]	


def create_initial_pop(max_depth=2, node=None):
	if node == None:
		node_type = get_node_type()
		if(node_type == 'rand'):
			node_type = randint(-5, 5)
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

		if(node_type == 'rand'):
			node_type = randint(-5, 5)

		if node.num_childs < node.arity:

			leaf = Tree(node_type)
			
			leaf.parent = node

			node.num_childs += 1
		return
	
	elif node.num_childs < node.arity:
		node.num_childs += 1
		node_type = get_node_type()

		if(node_type == 'rand'):
			node_type = randint(-5, 5)
		
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

		#print(leaf)
		return


def calculate_tree_value(tree, x):
	if(tree.name == 'sub'):
		return calculate_tree_value(tree.children[0], x) - calculate_tree_value(tree.children[1], x)
	if(tree.name == 'sum'):
		return calculate_tree_value(tree.children[0], x) + calculate_tree_value(tree.children[1], x)
	if(tree.name == 'mul'):
		return calculate_tree_value(tree.children[0], x) * calculate_tree_value(tree.children[1], x)
	if(tree.name == 'div'):
		dividendo = calculate_tree_value(tree.children[1], x)
		if(dividendo >= -0.05 and dividendo <= 0.05):
			dividendo = 1
		return calculate_tree_value(tree.children[0], x) / dividendo
	if(tree.name == 'x0'):
		#retornar valor de x0
		return x[0]
	if(tree.name == 'x1'):
		#retornar valor x1
		return x[1]
	if(tree.name in terminals):
		return int(tree.name)
	pass


def calculate_fitness(tree, x, y):
	ind_fitness = 0
	for i in range(0, len(y)):
		ind_fitness += (y[i] - calculate_tree_value(tree, x[i])) * (y[i] - calculate_tree_value(tree, x[i]) )

	ind_fitness = sqrt(ind_fitness)

	print(ind_fitness)

	#fitness = y - tree_value
	#print(abs(fitness))
	return ind_fitness


def tournament_selection(pop, k=2):
	(best, fitness) = (None, None)
	for i in range(0, k):
		ind = pop[randint(0, len(pop)-1)]
		if best == None or fitness > ind[1]:
			best = ind[0]
			fitness = ind[1]

	print("Fit: ", fitness)
	print(best)
	return best


def crossover(parent_1, parent_2):
	print(">>>>>>>>>>>CROSSOVER<<<<<<<<<<<<")
	print("\n\n")

	print(parent_1)

	print(parent_2)


	crossover_1_height = int(parent_1.height/3)
	crossover_2_height = int(parent_2.height/3)
	#print("Cross Height: ", parent_1.height)

	tree_1 = None
	tree_2 = None

	for pre, fill, node in RenderTree(parent_1):
		print(pre, node.name)

	for pre, fill, node in RenderTree(parent_2):
		print(pre, node.name)

		
	print("Parent")
	print(parent_1)

	for pre, fill, node in RenderTree(parent_1):
		for child in node.children:
			print("Name: ", child.name)
			#node = parent_2
			#parent_2.parent = node.parent
			break
	
	print("New parent")
	print(parent_1)

	# for pre, fill, node in RenderTree(parent_2):
	# 	if(node.height == crossover_2_height):
	# 		tree_2 = node
	# 		break

	# print("Tree_1: ", tree_1)
	# for pre, fill, node in RenderTree(tree_1):
	# 	print(pre, node.name)
	# print("Tree_2: ", tree_2)
	# for pre, fill, node in RenderTree(tree_2):
	# 	print(pre, node.name)
	
def printtree(tree_node):
    if tree_node.left is not None:
        printtree(tree_node.left)
    print("Name: ", tree_node.name, " Depth: ", tree_node.depth)
    if tree_node.right is not None:
        printtree(tree_node.right)


fileLoad = FileLoad(dataset_path)
x, y = fileLoad.getData()

for i in range(2000):
	tree = create_initial_pop()

	#printtree(tree)
	#print(tree)

	#tree_value = calculate_tree_value(tree, x)
	#print("Value: ", tree_value)

	fitness = calculate_fitness(tree, x, y)

	pop_fitness.append( (tree, fitness) )

	#print("Pop: ", pop_fitness)
	#continue



#Calculate children population
child_population = []
while len(child_population) < gen_size:
	best_1 = tournament_selection(pop_fitness, k=200)
	best_2 = tournament_selection(pop_fitness, k=200)

	#crossover(best_1, best_2)

	break



