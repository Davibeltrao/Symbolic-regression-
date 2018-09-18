from anytree import Node, RenderTree
from random import randint
import os
import csv
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


gen_size = 50
terminals = ['x0', 'x1', 3, 5]
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
		
		return X_return, y_return

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
		#print(node_type)

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
		#print("Depth: ", node.depth, " Arity: ", node.arity)
		
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
		if(dividendo >= -1 and dividendo <= 1):
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


def calculate_fitness(tree_value, y):
	fitness = y - tree_value
	print(abs(fitness))
	return fitness


def crossover(parent_1, parent_2):
	print()
	print()

	crossover_1_height = int(parent_1.height/3)
	crossover_2_height = int(parent_2.height/3)
	print("Cross Height: ", crossover_2_height)

	tree_1 = None
	tree_2 = None

	for pre, fill, node in RenderTree(parent_1):
		print(pre, node.name)

	for pre, fill, node in RenderTree(parent_2):
		print(pre, node.name)

	for pre, fill, node in RenderTree(parent_1):
		if(node.height == crossover_1_height):
			tree_1 = node
			break
		
	for pre, fill, node in RenderTree(parent_2):
		if(node.height == crossover_2_height):
			tree_2 = node
			break
	

	print("Tree_1: ", tree_1)
	for pre, fill, node in RenderTree(tree_1):
		print(pre, node.name)
	print("Tree_2: ", tree_2)
	for pre, fill, node in RenderTree(tree_2):
		print(pre, node.name)
	

fileLoad = FileLoad(dataset_path)

for i in range(gen_size):
	x, y = fileLoad.getData()

	tree = create_initial_pop()
	
	for pre, fill, node in RenderTree(tree):
	#	print(pre, node.name)
		pass

	tree_value = calculate_tree_value(tree, x)
	
	fitness = calculate_fitness(tree_value, y)

	pop_fitness.append( (tree, fitness ))

print(pop_fitness)

crossover(pop_fitness[0][0], pop_fitness[1][0])



