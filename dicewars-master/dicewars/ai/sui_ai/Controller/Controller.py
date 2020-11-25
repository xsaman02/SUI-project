

class Controller():
	"""
	Class for easier usage of KNN and UCS
	"""
	
	def __init__(self, KNN, UCS) -> None:
		self.knn = KNN
		self.ucs = UCS