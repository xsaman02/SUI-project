import numpy as np
import json, math


def plot(dps, cs):
	plt.figure()
	# dps = knn.dataset
	# cs = [knn.get_class_of_datapoint(dp) for dp in dps]

	pos_dps, neg_dps, undec_dps = [], [], []
	for d, c in zip(dps, cs):
		if c:
			pos_dps.append(d)
		elif c == False:
			neg_dps.append(d)
		else:
			undec_dps.append(d)

	pos_dps = np.asarray(pos_dps).transpose()
	neg_dps = np.asarray(neg_dps).transpose()
	undec_dps = np.asarray(undec_dps).transpose()
	show = False
	if len(pos_dps) != 0:
		plt.scatter(pos_dps[0], pos_dps[1], color="green")
		show = True

	if len(neg_dps) != 0:
		plt.scatter(neg_dps[0], neg_dps[1], color="orange")
		show = True		

	if len(undec_dps) != 0:
		plt.scatter(undec_dps[0], undec_dps[1], color="gray")
		show = True	

	if show:
		plt.show()
	


class KNN():
	
	dataset_addr = "./datapoints.save"
	hashtable_addr = "./hashtable.save.json"
	dataset = np.asarray([])
	deviation = 0.0
	hashtable = {}
	
	def __init__(self, k : int) -> None:
		"""Init function\n
	    Parameters:
    	----------
		k (int): k sets number of nearest searching datapoints
		"""
		self.k = k


	def initialize(self, params : dict, n_data : int) -> None:
		"""		Function initialize new dataset via Monte Carlo.\n		
	    Parameters:
    	----------
		params (dict): key = name of feature, value = array (min_val, max_val)
		ndata (int): number of generated datapoints
		"""
		self.dataset = np.random.rand(n_data, len(params.keys()))
		self.min_max = list(params.values())
		
	def first_mapping(self, pos_datapoint, neg_datapoint, undecided=None) -> None:
		"""	This function is called after new inicialization.\n
		Starts mapping new datapoints from dataset to classes based on given datapoints.\n\n
		Given points will be included in dataset.\n
		Classes of datapoint will be hashed for later quicker search

		Parameters:
		-----------
			pos-datapoint (np.array): positive datapoint
			neg-datapoint (np.array): negative datapoint
			undecided (np.array): help parameter for recursive evaluation
		"""
		assert(type(self.min_max) != type(None))
		threshold = 0.5

		if type(undecided) == type(None):
			self.set_new_data(pos_datapoint, True)
			self.set_new_data(neg_datapoint, False)
			dataset = self.dataset
		else:
			dataset = undecided
		
		undecided = np.array([])
		for dp in dataset:
			if dp.data.tobytes not in self.hashtable:
				r = self.__evaluate(dp)
				if r == None:
					undecided = np.array([dp]) if undecided.size == 0 else np.append(undecided, [dp], axis=0)
				else:
					self.hashtable[dp.data.tobytes] = r > threshold
		if undecided.size == 0:
			print(len([x for x in self.hashtable.values() if x == True]))
			print(len([x for x in self.hashtable.values() if x == False]))
			return
		else:
			print("undecided", undecided.shape[0])
			plot(self.dataset, [self.__get_class_of_datapoint(dp) for dp in self.dataset])
			self.first_mapping(None, None, undecided)

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
		self.dataset = np.loadtxt(self.dataset_addr)
		with open(self.hashtable_addr, "r+") as fd:
			self.hashtable = json.load(fd)
		return self.dataset

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
		assert(self.dataset.size != 0)
		
		datapoints = self.__get_k_neighbors(self.__normalize(datapoint))
		classes = np.array([self.__get_class_of_datapoint(dp[0]) for dp in datapoints])
		pos = np.count_nonzero(classes == True)
		neg = np.count_nonzero(classes == False)
		if pos + neg == 0:
			return None
		return pos / (pos+neg)


	def __evaluate(self, datapoint):
		"""help function for evaluation od data, that has been already normalized

		Parameters:
		-----------
			datapoint (np.array()): datapoint (vector of features)

		Returns:
		--------
			float: index of recommended attack
		"""
		assert(self.dataset.size != 0)
		
		datapoints = self.__get_k_neighbors(datapoint)
		classes = np.array([self.__get_class_of_datapoint(dp[0]) for dp in datapoints])
		pos = np.count_nonzero(classes == True)
		neg = np.count_nonzero(classes == False)
		if pos + neg == 0:
			return None
		return pos / (pos+neg)


	def set_new_data(self, data, y : bool) -> None:
		"""	Function adds new data (vector of features) to the dataset and marked it by class y (0, 1)

		Parameters:
		-----------
			data (numpy.array): vector of features
			y (bool): class
		"""
		assert(self.dataset.size != 0)
		data = self.__normalize(data)
		self.dataset = np.append(self.dataset, [data], axis=0)
		self.hashtable[data.data.tobytes()] = y


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

	def save_dataset(self) -> None:
		"""Saves loaded dataset to hard-coded addr
		"""
		np.savetxt(self.dataset_addr, self.dataset)
		with open(self.hashtable_addr, "w") as fd:
			json.dump(self.hashtable, fd)

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

	def __get_k_neighbors(self, datapoint, deviation=None, selected_datapoints=np.array([]), last_deviation=None, recur_count=0) -> np.ndarray:
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
		

		assert(self.deviation > 0)

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


		datapoints = np.array([v for v in v_b if v in v_s and v.data.tobytes() in self.hashtable])


		if self.k <= len(datapoints) or recur_count >= 2:
			if len(datapoints) <= 2*self.k or recur_count >= 2:
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
		# show = False
		# if list(dp) == [0.8, 0.75]:
		# 	show = True
		dp = dp.data.tobytes()
		if dp in self.hashtable:
			# if show:
			# 	print("positive dp "+dp+" class: ",self.hashtable[dp])
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

		assert(type(self.min_max) != type(None))
		assert(len(self.min_max) == len(v))

		for i in range(v.size):
			v[i] = (v[i] - self.min_max[i][0]) / (self.min_max[i][1] - self.min_max[i][0])
		return np.array(v)




import time
import matplotlib.pyplot as plt
if __name__ == "__main__":
	knn = KNN(3)
	d = {"probability of capture" : [0, 10], "probability of sustain" : [0,1]}
	knn.initialize(d, 10)
	knn.set_deviation(0.1)
	pos = np.array([8, 0.75])
	neg = np.array([1, 0.2])
	print("starts mapping")
	knn.first_mapping(pos, neg)
	print("init completed")

