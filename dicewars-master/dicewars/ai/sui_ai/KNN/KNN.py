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
	deviation = 0.1
	hashtable = {}
	
	def __init__(self, k : int, min_max : Iterator[List[float]]) -> None:
		"""Init function\n
	    Parameters:
    	----------
		k (int): k sets number of nearest searching datapoints
		min_max ([ [float | int, float | int] ]) : minimum and maximum values of features in datapoint
		"""
		self.k = k
		self.min_max = min_max


	def initialize(self, n_data, len_of_vector : int) -> None:
		"""		Function initialize new dataset via Monte Carlo.\n		
	    Parameters:
    	----------
		ndata (int): number of generated datapoints
		"""
		self.dataset = np.random.rand(n_data, len_of_vector)
		self.hashtable = {hash(bytes(data)) : 1 for data in self.dataset}

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
		print("loading dataset")
		self.dataset = np.loadtxt(f'{self.dataset_addr}{self.dataset_index}.save')
		print("dps loaded")
		with open(f'{self.hashtable_addr}{self.dataset_index}.save.csv', "r+") as fd:
			reader = csv.reader(fd)
			self.hashtable = {int(dp_bytes) : int(c) for dp_bytes, c in reader}
		print("hashtable loaded")

	def save_dataset(self) -> None:
		"""Saves loaded dataset to hard-coded addr
		"""
		np.savetxt(f'{self.dataset_addr}{self.dataset_index}.save', self.dataset)
		with open(f'{self.hashtable_addr}{self.dataset_index}.save.csv', "w") as fd:
			writer = csv.writer(fd)
			writer.writerows(self.hashtable.items())

	def evaluate(self, datapoint):
		"""		Function evaluate given datapoint (vector of features) on loaded dataset and returns index of recomended attacking (0-1)

	    Parameters:
    	----------
		datapoint (np.array): vector of features

		Returns:
		float: index of recomended attacking (0-1)
		None in case of undecided

		Important:
		----------
		If dataset and metrics is not set, will raise exception
		"""
		assert(self.dataset.size > 0)

		datapoints = self.__get_k_neighbors(self.__normalize(datapoint))
		print("datapoints len:", len(datapoints))
		classes = np.array([self.__get_class_of_datapoint(dp) for dp in datapoints])

		pos = (classes == 1).sum()
		neg = (classes == 0).sum()
		if pos + neg == 0:
			return None
		return pos / (pos + neg)

	def set_new_datapoint(self, data, y) -> None:
		"""	Function adds new data (vector of features) to the dataset and marked it by class y (0, 1)

		Parameters:
		-----------
			data (numpy.array): vector of features
			y (bool | int): class
		"""
		assert(self.dataset.size != 0)
		data = self.__normalize(data)
		self.dataset = np.append(self.dataset, [data], axis=0)
		self.hashtable[hash(bytes(data))] = int(y)


	def set_deviation(self, deviation) -> None:
		"""	Function takes deviation on which will start searching nearest neighbors\n
		
	    Parameters:
    	----------
		metrics (float): number from 0-1

		Example:
    	----------
		1D data:
			deviation: (0.25)
			data: (0.5)
			searching pool for nearest neighbors: (0.25 - 0.75)\n
		
		Info:
		----
		Metrics set up is good for cutting searching and computing time\n
		and is mandatory to set up before evaluating.\n
		"""
		self.deviation = deviation

	def __compute_euclid_dist(self, a, b) -> float:
		"""		Function takes vectors of two datapoints and compute euclid distance between them\n
		parameters: a, b both needs to be numpy arrays\n

		Parameters
		----------
		a (np.array): datapoint1
		b (np.array): datapoint2

		Returns:
		float: euclid distance
		"""
		return np.linalg.norm(b-a)

	def __get_k_neighbors(self, datapoint, deviation=None, selected_datapoints=np.array([]), last_deviation=None, recur_count=0, max_recur_count=4) -> np.ndarray:
		"""Get k nearest neighbors from given datapoint

		Parameters:
		-----------
			datapoint (np.array): Given datapoint
			deviation (np.array, optional): Custom metrics for setting up searching pool. Defaults to None.
			selected-datapoints (np.array, optional): Selected points for no need searching in whole dataset. Defaults to empty np.array.
			last-deviation (np.array): Help parametr for computing next step of deviation
		Returns:
		--------
			np.ndarray: Returns K nearest neighbors
		"""
		
		if deviation == None:
			deviation = self.deviation


		v_b, v_s = None, None
		if selected_datapoints.size != 0:
			v_b = np.all(selected_datapoints >= (datapoint - deviation), axis=1)
			v_s = np.all(selected_datapoints <= (datapoint + deviation), axis=1)
			
			i = np.argwhere(v_b==True)
			v_b = selected_datapoints[i]
			i = np.argwhere(v_s==True)
			v_s = selected_datapoints[i]
		else:
			v_b = np.all(self.dataset >= (datapoint - deviation), axis=1)
			v_s = np.all(self.dataset <= (datapoint + deviation), axis=1)
		
			i = np.argwhere(v_b==True)
			v_b = self.dataset[i]
			i = np.argwhere(v_s==True)
			v_s = self.dataset[i]


		datapoints = np.array([v for v in v_b if v in v_s])


		if self.k <= len(datapoints) or recur_count >= max_recur_count:
			if len(datapoints) <= 2*self.k or recur_count >= max_recur_count:
				if len(datapoints) <= self.k:
					return datapoints
				distances = []
				for d in datapoints:
					distances.append(self.__compute_euclid_dist(datapoint, d))
				k_min = np.argsort(distances)[0:self.k]
				return np.array(datapoints)[k_min]
			else:
				# Too much points 
				if last_deviation == None:
					difference = deviation / 2
				else:	
					difference = deviation - abs(deviation - last_deviation) * 0.5
				
				return self.__get_k_neighbors(datapoint, difference, np.asarray(selected_datapoints), last_deviation=deviation, recur_count=recur_count+1)
		else:
			# Too little points
			if last_deviation == None:
				difference = 2*deviation
				return self.__get_k_neighbors(datapoint, difference, np.asarray(selected_datapoints), last_deviation=None, recur_count=recur_count+1)
			else: 
				difference = deviation + abs(deviation - last_deviation) * 0.5
				return self.__get_k_neighbors(datapoint, difference, np.asarray(selected_datapoints), last_deviation=deviation, recur_count=recur_count+1)
	


	def __get_class_of_datapoint(self, dp) -> bool:
		"""	Function returns class of given datapoint\n
			If class is not known, returns None
		"""
		print("got in __get_class_of_datapoint")
		dp = hash(bytes(dp))
		if dp in self.hashtable:
			if dp == hash(bytes(np.array([0.5, 0.5, 0.5, 0.5]))):
				print("res: ", self.hashtable[dp])
			return self.hashtable[dp]
		return None

	def __normalize(self, v):
		"""Normalize given numpy vector by setup min_max intervals

		Parameters:
		-----------
			v (numpy.array()): Given vector (datapoint)

		Returns:
		--------
			numpy.array(): normalized vector
		"""

		print("got in __normalize")
		assert(type(self.min_max) != type(None))
		assert(len(self.min_max) == len(v))

		for i in range(v.size):
			v[i] = (v[i] - self.min_max[i][0]) / (self.min_max[i][1] - self.min_max[i][0])
		return np.array(v)

	def get_len_of_dataset(self) -> int:
		return self.dataset.shape[0]

	def create_new_dataset(self) -> None:
		self.save_dataset()
		self.dataset_index += 1
		self.initialize(100, len(self.min_max))



if __name__ == "__main__":
	import time
	d = {"probability of capture" : [0, 1], 
		 "change of biggest region size after attack" : [0,15], 
		 "mean dice of enemy terrs. of target" : [1, 8], 
		 "attacker dice" : [1, 8]}
	knn = KNN(11, list(d.values()))
	knn.initialize(d, 100)
	knn.set_new_datapoint(np.array([0.5,7.5,4.5,4.5]), False)
	knn.save_dataset()
	knn.load_dataset()
	knn.set_deviation(0.2)
	# knn.load_dataset()
	# knn.set_min_max_vals([d.values()])

	# print(hash(bytes(knn.__normalize(np.array([0.5,33,1,-2])))) in knn.hashtable)
	# print(hash(bytes(np.array([0.5,33,1,-2]))))
	# print(knn.hashtable.items())

	start = time.perf_counter()
	prob = knn.evaluate(np.array([0.6,7.5,3.5,3.5]))
	stop = time.perf_counter()
	print(stop-start)
	print(prob)
	# print(list(knn.hashtable.keys())[0])
	# print(knn.hashtable.items())


