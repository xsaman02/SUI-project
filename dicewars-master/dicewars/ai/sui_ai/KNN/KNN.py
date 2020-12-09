from os import write
from typing import Iterator, List
import numpy as np
import json, math, csv
from hashlib import sha1



class KNN():
	
	dataset_addr = "./dicewars/ai/sui_ai/KNN/datapoints"
	hashtable_addr = "./dicewars/ai/sui_ai/KNN/hashtable"
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

	def evaluate(self, datapoint):

		n_dp = self.__normalize(datapoint)
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

	def create_new_dataset(self) -> None:
		self.save_dataset()
		self.dataset_index += 1
		self.initialize(100, len(self.min_max))



if __name__ == "__main__":
	d = {"probability of capture" : [0, 1], 
		"change of biggest region size after attack" : [0,15], 
		"mean dice of enemy terrs. of target" : [0, 8], 
		"mean dice of enemy terrs. of source" : [1, 8]}
	knn = KNN(11, list(d.values()), np.array([1, 1.1, 1.2, 1.2]))
	knn.initialize(100, len(d.keys()))
	knn.save_dataset()







	# d = {"probability of capture" : [0, 1], 
	# 	 "change of biggest region size after attack" : [0,15], 
	# 	 "mean dice of enemy terrs. of target" : [1, 8], 
	# 	 "attacker dice" : [1, 8]}
	# knn = KNN(11, list(d.values()), np.array([1, 1.2, 1.4, 1.5]))
	# knn.initialize(1000, len(d.keys()))
	# dp = np.array([0.5,7.5,4.5,4.5])

	# knn.set_new_datapoint(, False)
	# knn.save_dataset()
	# exit(0)
	# knn.load_dataset()

	# start_old = time.perf_counter()
	# t = []
	# for _ in range(10000):
	# 	dp = np.random.random(4)
	# 	start = time.perf_counter()
	# 	prob = knn.evaluate(dp)
	# 	stop = time.perf_counter()
	# 	t.append(stop-start)
	# end_old = time.perf_counter()
	# print("old evaluation way: ", np.asarray(t).mean())
	# print("whole evaluation on 10000 samples cost: ", end_old-start_old, "s")



