
import copy


class TreeNode():
	"""	Help class for creating tree structure
	"""
	paths = []

	def __init__(self, attacker, player_name) -> None:
		self.attacker = attacker
		self.player_name = player_name
		self.possible_attacks = []

	def expand_to_n_level(self, n, dices, board, visited=[]):
		""" expand possible attacks to n level

		Args:
			lvl (int): maximum level of immersion to expand in search tree

		Returns:
			[int]: number of all attacks
		"""
		n_attacks = 0
		if n == 0 or dices <= 1 :
			return 0
		neigbours = [board.get_area(x) for x in board.get_area(self.attacker).get_adjacent_areas()]
		for neig in neigbours:
			if neig.get_owner_name() == self.player_name or neig in visited:
				continue
			else:
				self.possible_attacks.append(TreeNode(neig.get_name(), self.player_name))
				n_attacks += 1
				n_attacks += self.possible_attacks[-1].expand_to_n_level(n - 1, dices-1, board, visited + [self.attacker])

		# for subnode in self.possible_attacks:
		# 	n_attacks += subnode.expand_to_n_level(n - 1, dices-1, board, visited + [self.attacker])

		return n_attacks
	
	def get_paths_to_n_level(self, n, paths=[], cur_path=[]):
		if n <= 0:
			paths.append(cur_path)
			return paths

		if self.attacker not in cur_path:
			cur_path.append(self.attacker)
		
		if len(self.possible_attacks) <= 0:
			paths.append(cur_path)
			return paths

		loop_attacks = 0
		for subnode in self.possible_attacks:
			if subnode.attacker in cur_path:
				loop_attacks +=1
				continue
			paths = subnode.get_paths_to_n_level(n-1, paths, copy.deepcopy(cur_path))
		if loop_attacks == len(self.possible_attacks):
			paths.append(cur_path)
		return paths


	def expand_node(self, max_depth, attacker, board, cur_path = []):
		cur_path.append(attacker.get_name())
		if max_depth <= 1:
			self.paths.append(cur_path)
			return

		possible_attacks = [x for x in attacker.get_adjacent_areas() 
				   if board.get_area(x).get_owner_name() != self.player_name 
				   and x not in cur_path]

		if len(possible_attacks) <= 0:
			self.paths.append(cur_path)
			return

		for enemy in possible_attacks:
			self.expand_node(max_depth - 1, board.get_area(enemy), board, copy.deepcopy(cur_path))
		

	def clear(self):
		self.attacker = None
		self.possible_attacks = []
