

class Attack():
	""" Help structure for storing evaluated attacks
	"""
	def __init__(self, attack, result):
		""" Init function takes given parameters and store them

		Parameters:
			attack ( List(Area, Area) ): Attack from attacker to target
			result (float): result of classifier
		"""

		self.attack = attack
		self.result = result