import numpy
from dicewars.ai.template_ai.KNN import KNN
from dicewars.ai.template_ai.TreeSearch import TreeSearch

class Controller():
	"""	Controller class is used for easy mainteining KNN and TreeSearch sub-classes
	"""
	knn = None
	tree_search = None

	def __init__(self, k) -> None:
		"""Init function for declaring needed sub-classes

		Parameters:
		-----------
			k (int): parameter for declaring number of nearest searched neigbors
		"""
		self.knn = KNN(k)
		self.tree_search = TreeSearch()
