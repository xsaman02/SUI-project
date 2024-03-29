from typing import Iterator, List
import numpy as np
import csv



class KNN():
	
	dataset_addr = "./dicewars/ai/xsaman02/KNN/datapoints"
	hashtable_addr = "./dicewars/ai/xsaman02/KNN/hashtable"
	dataset_index = 0
	dataset = np.asarray([])
	hashtable = {}
	
	def __init__(self, k : int, min_max : Iterator[List[float]], metric : Iterator[float]) -> None:
		"""Init function\n
	    Parameters:
    	----------
		k (int): k sets number of nearest searching datapoints
		min_max ([ [float | int, float | int] ]) : minimum and maximum values of features in datapoint
		metric ([ float ]) : priority of feature
		"""
		self.k = k
		self.min_max = min_max
		self.metric = metric


	def initialize(self, n_data, len_of_vector : int) -> None:
		"""		Function initialize new dataset via Monte Carlo.\n		
	    Parameters:
    	----------
		ndata (int): number of generated datapoints
		"""
		self.dataset = np.random.rand(n_data, len_of_vector)
		for i, metric in enumerate(self.metric):
			self.dataset[i] *= metric
		self.hashtable = {tuple(data) : 1 for data in self.dataset}

	def set_min_max_vals(self, min_max : list) -> None:
		"""setting intervals for input data, which is used for normalization

		Parameters:
		-----------
			min_max (list): 2D array - [[min0, max0], [min1, max1], ...]
		"""
		self.min_max = min_max

	def load_dataset(self) -> None:
		"""Function load dataset and hashtable from hardcoded addreses
		"""
		self.dataset = np.loadtxt(f'{self.dataset_addr}{self.dataset_index}.save')
		with open(f'{self.hashtable_addr}{self.dataset_index}.save.csv', "r+") as fd:
			reader = csv.reader(fd)
			self.hashtable = {tuple(eval(dp_bytes)) : float(c) for dp_bytes, c in reader}

	def save_dataset(self) -> None:
		"""Saves loaded dataset to hard-coded addr
		"""
		np.savetxt(f'{self.dataset_addr}{self.dataset_index}.save', self.dataset)
		with open(f'{self.hashtable_addr}{self.dataset_index}.save.csv', "w") as fd:
			writer = csv.writer(fd)
			writer.writerows(self.hashtable.items())

	def evaluate(self, dp):
		""" Evaluates given datapoint 
			and returns mean from K nearest neigbours in dataset

		Args:
		-----
			dp (np.ndarray): vector of features

		Returns:
		--------
			float : mean of K nearest neigbours in dataset
		"""

		n_dp = self.__normalize(dp)
		neigbours = self.__get_k_neighbors(n_dp)
		classes = np.array([self.__get_class_of_datapoint(dp) for dp in neigbours])

		return np.array(classes).mean()

	def set_new_datapoint(self, data, y) -> None:
		"""	Function adds new data (vector of features) to the dataset and marked it by class y (0, 1)

		Parameters:
		-----------
			data (numpy.array): vector of features
			y (bool | int): class
		"""

		data = self.__normalize(data)
		self.dataset = np.append(self.dataset, [data], axis=0)
		self.hashtable[tuple(data)] = y
		# print("dataset shape = ", self.dataset.shape)


	def __get_k_neighbors(self, dp):
		""" Gets K nearest neigbours in dataset to given dp

		Args:
		-----
			dp (np.ndarray): datapoint

		Returns:
		--------
			np.ndarray: 2D-matrix - K nearest neigbours
		"""
		lens = np.linalg.norm(self.dataset - dp, axis=1)
		indecies = np.argsort(lens)[:self.k]
		return self.dataset[indecies]

	def __get_class_of_datapoint(self, dp) -> bool:
		"""	Function returns class of given datapoint\n
			class can by 0, 0.5, 1
		"""
		return self.hashtable[tuple(dp)]

	def __normalize(self, v):
		"""Normalize given numpy vector by setup min_max intervals

		Parameters:
		-----------
			v (numpy.array()): Given vector (datapoint)

		Returns:
		--------
			numpy.array(): normalized vector
		"""

		for i in range(v.size):
			v[i] = (v[i] - self.min_max[i][0]) / (self.min_max[i][1] - self.min_max[i][0]) * self.metric[i]
		return np.array(v)

	def get_len_of_dataset(self) -> int:
		return self.dataset.shape[0]


if __name__ == "__main__":
	"""	Execute this program in case of need to initialize new dataset. 
	"""
	d = {"probability of capture" : [0, 1], 
		"change of biggest region size after attack" : [0,8], 
		"mean dice of enemy terrs. of target" : [0, 8], 
		"mean dice of enemy terrs. of source" : [1, 8]}
	knn = KNN(11, list(d.values()), np.array([1, 1, 1, 1]))

	
	
	# Set new dataset with given number of datapoint
	knn.initialize(100, len(d.keys()))
	knn.save_dataset()

	# knn.load_dataset()
	# knn.set_new_datapoint(np.array([0.45, 1, 0, 1]), 1)
	# knn.set_new_datapoint(np.array([0.46, 1, 0, 1]), 1)
	# knn.save_dataset()


	exit(0)
	
	
	knn.load_dataset()
	hashtable = knn.hashtable
	from mpl_toolkits import mplot3d
	from matplotlib import pyplot as plt

	arr_zero = []
	arr_half = []
	arr_full = []
	for dp, val in hashtable.items():
		if val == 0:
			arr_zero.append(dp)
		elif val == 0.5:
			arr_half.append(dp)
		else:
			arr_full.append(dp)

	print("lenghts: zeros: {}, half: {}, ones: {}".format(arr_zero, arr_half, arr_full))

	x_zero = [x[0] for x in arr_zero]
	x_half = [x[0] for x in arr_half]
	x_full = [x[0] for x in arr_full]

	y_zero = [y[2] for y in arr_zero]
	y_half = [y[2] for y in arr_half]
	y_full = [y[2] for y in arr_full]

	z_zero = [z[3] for z in arr_zero]
	z_half = [z[3] for z in arr_half]
	z_full = [z[3] for z in arr_full]


	# Creating figure
	fig = plt.figure(figsize = (10, 7))
	ax = plt.axes(projection ="3d")
	plt.xlabel("prob. to conquer")
	plt.ylabel("enemy dice mean of target")
	# plt.zlabel("enemy dice mean of source")

	# Creating plot
	ax.scatter3D(x_zero, y_zero, z_zero, color = "red")
	ax.scatter3D(x_half, y_half, z_half, color = "orange")
	ax.scatter3D(x_full, y_full, z_full, color = "green")


	# show plot
	plt.show()

