from anytree import Node, RenderTree
import os
import csv
import numpy as np
from math import sqrt
import math
import copy
import random
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
	
pop_list = []
pop_fitness = []

#max_depth = 1

tree_id = 0


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
	tree_id = 0
	arity = 0 # 0 = Terminal / 1 = (exp, cos, sin) / 2 = (mul, div, sub, sum)
	def __init__(self, name):
		super().__init__(name)
				
	def isMaxDepth(self):
		if(self.depth >= 6):
			return True
		return False


def get_node_type(terminal_only=False, first=False):
	node_type = random.randint(0, 15)
	#print("type: ",node_type)
	if (node_type == 0 or terminal_only==True) and not first:
		terminal_node = random.randint(0, len(terminals)-1)
		return terminals[terminal_node]
	else:
		function_node = random.randint(0, len(functions)-1)
		return functions[function_node]	


def create_initial_pop(max_depth=5, node=None):
	global tree_id
	if node == None:
		node_type = get_node_type(first=True)
		if(node_type == 'rand'):
			node_type = randint(-5, 5)
		#create root node
		root = Tree(node_type)
		root.tree_id = tree_id
		tree_id += 1	
		#get arity for root oeprator
		if node_type in terminals:
			root.arity = 0
		elif node_type in ['sin', 'cos', 'exp', 'log']:
			root.arity = 1
			#criar um filho recursivamente
			create_initial_pop(max_depth, node=root)
		else:
			root.arity = 2
		
			#criar dois filhos recursivamente
			create_initial_pop(max_depth, node=root)
			create_initial_pop(max_depth, node=root)
		
		#pop_list.append(root)
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
			leaf.tree_id = tree_id
			tree_id += 1
			
			leaf.parent = node

			node.num_childs += 1
		return
	
	elif node.num_childs < node.arity:
		node.num_childs += 1
		node_type = get_node_type()

		if(node_type == 'rand'):
			node_type = randint(-5, 5)
		
		leaf = Tree(node_type)
		leaf.tree_id = tree_id
		tree_id += 1
				
		leaf.parent = node

		#get arity for leaf operator
		if node_type in terminals:
			leaf.arity = 0
		elif node_type in ['sin', 'cos', 'exp', 'log']:
			leaf.arity = 1
			#criar somente um filho(arity 1)
			create_initial_pop(max_depth, node=leaf)
		else:
			leaf.arity = 2
			#chamar recursivamente para criar dois nós filhos(arity 2)
			create_initial_pop(max_depth, node=leaf)
			create_initial_pop(max_depth, node=leaf)

		#print(leaf)
		return


def calculate_tree_value(tree, x):
	try:
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
		if(tree.name =='log'):
			log = calculate_tree_value(tree.children[0], x)
			if log <= 1:
				log = 2 
			return math.log(log, 2)
		
		if(tree.name =='sin'):
			return math.sin(calculate_tree_value(tree.children[0], x))

		if(tree.name =='cos'):
			return math.cos(calculate_tree_value(tree.children[0], x))	

		if(tree.name =='sqrt'):
			number = calculate_tree_value(tree.children[0], x)
			if number <= 0:
				return number
			else:
				return math.sqrt(number)	

		if(tree.name =='pow'):
			return calculate_tree_value(tree.children[0], x) * calculate_tree_value(tree.children[0], x)

		if(tree.name == 'x0'):
			#retornar valor de x0
			return x[0]
		if(tree.name == 'x1'):
			#retornar valor x1
			return x[1]
		if(tree.name == 'x2'):
			return x[2]
		if(tree.name == 'x3'):
			return x[3]
		if(tree.name == 'x4'):
			return x[4]
		if(tree.name == 'x5'):
			return x[5]
		if(tree.name == 'x6'):
			return x[6]
		if(tree.name == 'x7'):
			return x[7]
		if(tree.name in terminals):
			return int(tree.name)
	except:
		#print("Fucked")
		pass
	pass


def calculate_fitness(tree, x, y):
	ind_fitness = 0
	for i in range(0, len(y)):
		ind_fitness += (y[i] - calculate_tree_value(tree, x[i])) * (y[i] - calculate_tree_value(tree, x[i]))

	#ind_fitness = ind_fitness/len(y)

	y_mean = np.mean(y)

	bottom_value = 0

	for yi in y:
		bottom_value += (yi - y_mean) * (yi - y_mean)

	ind_fitness = ind_fitness / bottom_value

	ind_fitness = sqrt(ind_fitness)

	
	#print(ind_fitness)

	#fitness = y - tree_value
	#print(abs(fitness))
	return ind_fitness


def tournament_selection(pop, k=2):
	(best, fitness) = (None, None)
	for i in range(0, k):
		ind = pop[random.randint(0, len(pop)-1)]
		if best == None or fitness > ind[1]:
			best = ind[0]
			fitness = ind[1]

	#print("Fit: ", fitness)
	#print(best)
	return best, fitness


def split_tree(parent, cut_height=None):
	tree_list = []
	tree_1 = None
	tree_2 = None

	random_depth = random.randint(1, 2)

	cut_depth = parent.height-random_depth

	#print("Cut depth: ", cut_depth)

	#print("Height Parent: ", parent.height)

	# for pre, fill, node in RenderTree(parent):
	# 	if node.depth == cut_depth and tree_1 == None:
	# 		tree_1 = node
	# 		node.parent = None
	# 		tree_list.append(node)
		

	# for pre, fill, node in RenderTree(parent):
	# 	if node.depth == cut_depth and tree_2 == None and tree_1 != None:
	# 		tree_2 = node
	# 		node.parent = None

	for pre, fill, node in RenderTree(parent):
		if node.depth == cut_depth:
			tree_list.append(node)
	
	if len(tree_list) == 0:
		print("Couldn't extract trees in crossover")

	# for pre, fill, node in RenderTree(parent):
	# 	if node.depth == cut_depth and tree_2 == None and tree_1 != None:
	# 		tree_2 = node
	# 		node.parent = None


	# print("List: ", tree_list)
	# print("List len:", len(tree_list))

	tree_1, tree_2 = np.random.choice(tree_list, 2)
	tree_1.parent = None
	tree_2.parent = None
	
	# print("Split 1")
	# for pre, fill, node in RenderTree(tree_1):
	# 	print(node.name)

	# print('Split 2')
	# for pre, fill, node in RenderTree(tree_2):
	# 	print(node.name)

	return tree_1, tree_2, cut_depth

def crossover(parent_1, parent_2):
	# print(">>>>>>>>>>>CROSSOVER<<<<<<<<<<<<")
	# print("\n\n")

	# print(parent_1)
	# print(parent_2)

	tree_test = parent_1
	tree_test2 = parent_2


	crossover_1_height = int(parent_1.height/3)
	crossover_2_height = int(parent_2.height/3)
	#print("Cross Height: ", parent_1.height)
	#print("Cross Height: ", parent_2.height)

	if parent_2.height <= 1 or parent_1.height <= 1:
		#print("Returning crossover. Small individual size")
		return parent_1

	tree_1 = None
	tree_2 = None

	tree_3 = None
	tree_4 = None


	# print("Parent 1")
	# for pre, fill, node in RenderTree(tree_test):
	# 	print(pre, node.name)

	# print('Parent 2')
	# for pre, fill, node in RenderTree(tree_test2):
	# 	print(pre, node.name)

	#if tree_test.arity == 2:	
	tree_1, tree_2, cut_depth_1 = split_tree(tree_test)

	#if tree_test2.arity == 2:
	tree_3, tree_4, cut_depth_2 = split_tree(tree_test2)
	
	# print("Arvore a ser substituida")
	# for pre, fill, node in RenderTree(tree_test):
	# 	print(pre, node.name)

	for pre, fill, node in RenderTree(tree_test):
		if  len(node.children) < node.arity: 
			tree_3.parent = node
		
	for pre, fill, node in RenderTree(tree_test):
		if len(node.children) < node.arity: 
			tree_1.parent = node


	# print("End of crossover")
	# for pre, fill, node in RenderTree(tree_test):
	# 	print(pre, node.name)

	return tree_test

def mutation(tree):
	subtree = create_initial_pop(max_depth=3)

	tree_list = []
	cut_depth = tree.height-3

	for pre, fill, node in RenderTree(tree):
		if node.depth == cut_depth:
			tree_list.append(node)

	#print("Tree List: ", tree_list)

	try:
		tree_1 = random.choice(tree_list)
	#	print("tree_1: ", tree_1)
		tree_1.parent = None
	except:
		return tree
	#tree_2.parent = None

	# print('SubTree')
	# for pre, fill, node in RenderTree(subtree):
	# 	print(node.name)

	# print('Tree')
	# for pre, fill, node in RenderTree(tree):
	# 	print(node.name)

	for pre, fill, node in RenderTree(tree):
		if len(node.children) < node.arity:
			subtree.parent = node

	return tree
	
def printtree(tree_node):
    if tree_node.left is not None:
        printtree(tree_node.left)
    print("Name: ", tree_node.name, " Depth: ", tree_node.depth)
    if tree_node.right is not None:
        printtree(tree_node.right)


def plot_graph2(x, y, tree):
	def f(x, y):
		z_value = []
		for x_min in x:
			for y_un in y:
				z_value_step = []
				for x_un in x_min:
					#print("Pair: ", x_un, " ", y_un[0])
					calculate_x = [x_un, y_un[0]]
					value = calculate_tree_value(tree, calculate_x)
					#print("value: ", value)
					z_value_step.append(value)
				z_value.append(z_value_step)
			return z_value

	print("Shape: ", x)
	x0 = np.array([x_val[0] for x_val in x])
	x1 = np.array([x_val[1] for x_val in x])
	print("Shape0: ", x0.shape)
	print("Shape1: ", x1.shape)
	
	X, Y = np.meshgrid(x0, x1)
	z = f(X, Y)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.contour3D(x0, x1, z, 50, cmap='binary')

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	plt.show()


def plot_graph(x, y, tree):
	def f(x, y):
		z_value = []
		for x_min in x:
			for y_un in y:
				z_value_step = []
				for x_un in x_min:
					#print("Pair: ", x_un, " ", y_un[0])
					calculate_x = [x_un, y_un[0]]
					value = calculate_tree_value(tree, calculate_x)
					#print("value: ", value)
					z_value_step.append(value)
				z_value.append(z_value_step)
			return z_value
			break
		#return calculate_tree_value2(tree, x, y)
	
	fig = plt.figure()
	ax = plt.axes(projection='3d')	
	
	x = np.arange(-3, 3)
	print("x: ", x.shape)
	print("y: ", len(y))
	y = np.arange(-3, 3)
	print("y2: ", y.shape)
	#print("y: ", y)
	
	X, Y = np.meshgrid(x, y)
	#print("New x: ", X)
	#print("New y: ", Y)

	#print("Tree: ", tree)

	Z = f(X, Y)
	#print("Z: ", Z)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.contour3D(x, y, Z, 50, cmap='binary')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	
	plt.show()

def generate_initial_pop():
	for i in range(gen_size):
		tree = create_initial_pop(max_depth=pop_depth)

		fitness = calculate_fitness(tree, x, y)

		pop_fitness.append( (tree, fitness) )

def symbolic_regression(gen_cont, pop_fitness, elitism=True, file=None):
	best_total = None
	best_fit_total = None

	while gen_cont > 0:
		print('Generation: ', gen_cont)
		child_population = []
		fitness_list = []
		while len(child_population) < gen_size:
			best_1, fit_1 = tournament_selection(pop_fitness, k=2)
			#best_2, fit_2 = tournament_selection(pop_fitness, k=3)
			different_ind = False
			cont = 0
			
			while different_ind == False:
				best_2, fit_2 = tournament_selection(pop_fitness, k=2)
				cont += 1
				if best_2.tree_id != best_1.tree_id or cont > 15:
					different_ind = True
			
			first_ind = copy.deepcopy(best_1)
			second_ind = copy.deepcopy(best_2)

			new_ind = first_ind

			crossover_chance = random.randint(0, 100)
			#print("Cross chance: ", crossover_chance)
			if crossover_chance <= 90:
				#print("Entrei crossover")
				new_ind = crossover(first_ind, second_ind)		
			#	pass
			mutation_chance = random.randint(0, 100)
			#print("Mutation chance: ", mutation_chance)
			if mutation_chance <= 10 or fit_1 == fit_2:
				#print("Entrei mutation")
				new_ind = mutation(new_ind)

			new_ind.tree_id += 1

			fitness = calculate_fitness(new_ind, x, y)


			if elitism:
				if fit_1 < fitness and fit_1 <= fit_2:
					child_population.append( (best_1, fit_1) )
					fitness_list.append(fit_1)
				elif fit_2 < fitness and fit_2 <= fit_1:
					child_population.append( (best_2, fit_2) )
					fitness_list.append(fit_2)
				else:
					child_population.append( (new_ind, fitness) )
					fitness_list.append(fitness)
			else:
				child_population.append( (new_ind, fitness) )
				fitness_list.append(fitness)


		worst_fitness = np.amax(fitness_list)
		best_fitness = np.amin(fitness_list)
		average_fitness = np.average(fitness_list)

		for tree, fitness in child_population:
			if fitness == best_fitness:
				best_ind = tree

		print("Best: ", best_fitness, " Worst: ", worst_fitness, " Average: ", average_fitness)

		f.write("Gen: %d\n" % gen_cont)
		f.write("Best: " + str(best_fitness) + "  Worst: " + str(worst_fitness) + " Average: " + str(average_fitness) + "\n")

		pop_fitness = []
		pop_fitness = copy.deepcopy(child_population)

		fitness_child_repetition = []
		
		repetition = np.unique(fitness_list, return_counts=True)[1]
		
		count = sum([x-1 for x in repetition])

		print("Repetition Count: ", count)

		if best_fit_total == None or best_fit_total < best_fitness:
			best_fit_total = best_fitness
			best_total = best_ind

		gen_cont -= 1
		
	return best_total

random.seed(3443)
np.random.seed(3443)
	
#Features
gen_cont = 300
gen_size = 1500
pop_depth = 8
tournament_size = 2
terminals = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
functions = ['sum', 'sub', 'div', 'mul', 'sqrt', 'sin', 'cos', 'log']
elitism = True
dataset_path = 'datasets/concrete/concrete-train.csv'
test_path = 'datasets/concrete/concrete-test.csv'
dataset = 'concrete'


fileLoad = FileLoad(dataset_path)
x, y = fileLoad.getData()

testLoad = FileLoad(test_path)
x_test, y_test = testLoad.getData()


#for parameter in np.arange(3, 10, 1):
	#pop_depth = parameter

file_name = "result/concrete/" + str(dataset) + "_" + "GenCount=" + str(gen_cont) + "_" + "GenSize=" + str(gen_size) + "_" + "PopDepth=" + str(pop_depth) + "_" + "TournamentSize" + str(tournament_size) + "_" + "elitism=" + str(elitism) + '.txt'
f = open(file_name, "w+")

print("Gen size: ", gen_size)

generate_initial_pop()

best_total = symbolic_regression(gen_cont, pop_fitness,elitism=elitism, file=f)

f.close()

	#break
	#print("Sai")
	#print(len(y_test))
	#print(x_test.shape)
	#plot_graph2(x_test, y_test, best_total)
	#break

"""

"""