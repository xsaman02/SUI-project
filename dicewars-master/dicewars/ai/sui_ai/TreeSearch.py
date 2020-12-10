import copy

class TreeSearch():
	"""
	Class used for finding ways to attack the enemy provinces
	"""
	def __init__(self, player_name) -> None:
		""" Init function
		"""
		self.paths = []
		self.player_name = player_name

	def get_paths(self, area, depth_restrict, board):
		self.paths = []
		self.__expand_node(area.get_dice(), area, board, depth_restrict, cur_path=[])
		return self.paths


	def __expand_node(self, max_depth, attacker, board, depth_restrict, cur_path = []):
		"""Recursive function for evaluating all the possible ways from given attacker

		Args:
		-----
			- `max_depth` (int): maximum depth to expand
			- attacker (Area): Area of attacker
			- board (Board): Playing board
			- `depth_restrict` (int): maximum lenght of path
			- `cur_path` (list, optional): List to store current path evalution. Defaults to [].
		"""
		cur_path.append(attacker.get_name())

		if max_depth <= 1 or len(cur_path) >= depth_restrict:
			self.paths.append(cur_path)
			return

		possible_attacks = [x for x in attacker.get_adjacent_areas() 
				   if board.get_area(x).get_owner_name() != self.player_name 
				   and x not in cur_path]

		if len(possible_attacks) <= 0:
			self.paths.append(cur_path)
			return

		for enemy in possible_attacks:
			self.__expand_node(max_depth - 1, board.get_area(enemy), board,depth_restrict, copy.deepcopy(cur_path))
