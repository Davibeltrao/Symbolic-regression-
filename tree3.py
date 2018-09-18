from anytree import Node, RenderTree
from random import randint
import os
#import csv
import numpy as np
#from sklearn.metrics import mean_squared_error
from math import sqrt


gen_size = 50
terminals = ['x0', 'x1']
functions = ['sum', 'sub', 'div', 'mul']

pop_list = []
pop_list2 = []

pop_fitness = []

dataset_path = 'datasets/synth1/synth1-train.csv'

class Symbolic_regression:
	def __init__(self):
		pass

	def calculate_initial_pop(self):
		pass

	def calculate_tree_value(self):
		pass

	def calculate_fitness(self):
		pass

	def crossover(self):
		pass

	def tournament_selection(self):
		pass

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

class Node:
	arity = 0
	num_childs = 0
	depth = 0
	parent = None
	def __init__(self,val,left=None,right=None, children=None):
		self.name = val
		self.left = left
		self.right = right
		self.children = []
		self.height = 0

	def getVal(self):
		return self.name

	def setVal(self,newval):
		self.name = newval

	def setChild(self, child):
		self.children.append(child)

	def getLeft(self):
		return self.left

	def getRight(self):
		return self.right

	def setLeft(self,newleft):
		self.left = newleft

	def setRight(self,newright):
		self.right = newright

	def addHeight(self):
		self.height = self.height + 1

	def get_tree_height(self, root):	
		if root is None:
			return -1
		return max(self.get_tree_height(root.left), self.get_tree_height(root.right)) + 1

	def __str__(self, level=0):
		ret = "    "*level+repr(self.name)+"\n"
		for child in self.children:
			ret += child.__str__(level+1)
		return ret

	def __repr__(self):
		return '<tree node representation>'

def get_node_type(terminal_only=False):
	node_type = randint(0, 4)
	if node_type == 0 or terminal_only==True:
		terminal_node = randint(0, len(terminals)-1)
		return terminals[terminal_node]
	else:
		function_node = randint(0, len(functions)-1)
		return functions[function_node]	


def create_initial_pop(max_depth=7, node=None, node2=None, root=None):
	if node == None:
		node_type = get_node_type()
		
		#create root node
		root = Node(node_type)

		root2 = Tree(node_type)
		
		#get arity for root oeprator
		if node_type in terminals:
			root.arity = 0
		elif node_type in ['sin', 'cos', 'exp']:
			root.arity = 1
			#criar um filho recursivamente
			create_initial_pop(node=root, node2=root2, root=root)
		else:
			root.arity = 2
			#criar dois filhos recursivamente
			create_initial_pop(node=root, node2=root2, root=root)
			create_initial_pop(node=root, node2=root2, root=root)
		
		pop_list.append(root)
		pop_list2.append(root2)
		return root, root2
	
	elif node.depth == max_depth or node.arity == 0:
		print("Depth: ", node.depth, " Arity: ", node.arity)
		
		return

	elif node.depth == max_depth-1:
		node_type = get_node_type(terminal_only=True)
		if node.num_childs < node.arity:

			leaf = Node(node_type)
			leaf2 = Tree(node_type)
			
			node.setChild(leaf)
			leaf.parent = node

			leaf2.parent = node2

			if(node.left == None):
				node.setLeft(leaf)
			elif(node.right == None):
				node.setRight(leaf)

			node.num_childs += 1
			leaf.depth = node.depth + 1
		return
	
	elif node.num_childs < node.arity:
		node.num_childs += 1
		node_type = get_node_type()
		leaf = Node(node_type)
		leaf2 = Tree(node_type)

		
		leaf.depth = node.depth + 1
		
		node.setChild(leaf)
		leaf.parent = node

		leaf2.parent = node2

		if(node.getLeft() == None):
			node.setLeft(leaf)
		elif(node.getRight() == None):
			node.setRight(leaf)

		#get arity for leaf operator
		if node_type in terminals:
			leaf.arity = 0
		elif node_type in ['sin', 'cos', 'exp']:
			leaf.arity = 1
			#criar somente um filho(arity 1)
			create_initial_pop(node=leaf, node2=leaf2, root=root)
		else:
			leaf.arity = 2
			#chamar recursivamente para criar dois nos filhos(arity 2)
			create_initial_pop(node=leaf, node2=leaf2, root=root)
			create_initial_pop(node=leaf, node2=leaf2, root=root)

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


def calculate_fitness(tree_value, y):
	fitness = y - tree_value
	print(abs(fitness))
	return abs(fitness)


def crossover(parent_1, parent_2):
	print()
	print()

	height_1 = parent_1.get_tree_height(parent_1)
	height_2 = parent_2.get_tree_height(parent_2)

	crossover_1_height = int(height_1/2)
	crossover_2_height = int(height_2/2)
	print("Cross Height: ", crossover_2_height, crossover_1_height)

	child_1 = parent_1
	child_2 = parent_2

	print(child_1)

	tree_1 = getTree(child_1, crossover_1_height)
	print("Tree1")
	print(tree_1)

	tree_2 = getTree(child_2, crossover_2_height)
	print("Tree2")
	print(tree_2)

	tree_1 = None
	tree_2 = None


def getTree(tree, height):
	print(tree.depth)
	if tree.depth < height:
		for child in tree.children:
			return getTree(child, height)
	elif tree.get_tree_height(tree) >= 2:
		return tree

def tournament_selection(pop, k=2):
	(best, fitness) = (None, None)
	for i in range(0, k):
		ind = pop[randint(0, len(pop)-1)]
		if best == None or fitness > ind[1]:
			best = ind[0]
			fitness = ind[1]

	print("Fit: ", fitness)
	return best



fileLoad = FileLoad(dataset_path)

#Create initial population
for i in range(5):
	x, y = fileLoad.getData()

	tree, tree_vis = create_initial_pop()

	for pre, fill, node in RenderTree(tree_vis):
		print(pre, node.name)
		#pass

	print("Value: ", calculate_tree_value(tree, x))

	tree_value = calculate_tree_value(tree, x)

	fitness = calculate_fitness(tree_value, y)

	pop_fitness.append( (tree, fitness) )

	print("Pop: ", pop_fitness)


#Calculate children population
child_population = []
while len(child_population) < gen_size:
	best_1 = tournament_selection(pop_fitness, k=2)
	best_2 = tournament_selection(pop_fitness, k=2)

	crossover(best_1, best_2)

	break