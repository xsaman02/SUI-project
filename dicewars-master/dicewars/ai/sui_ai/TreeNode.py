

from dicewars.ai.utils import possible_attacks
from logging import PlaceHolder


class TreeNode():
	"""	Help class for creating tree structure
	"""

	def __init__(self, attacker, player_name) -> None:
		self.attacker = attacker
		self.player_name = player_name
		self.possible_attacks = []

	def expand_to_n_level(self, n):
		""" expand possible attacks to n level

		Args:
			lvl (int): level of immersion to expand in search tree

		Returns:
			[int]: number of all attacks
		"""
		if n == 0:
			return 0

		neigbours = self.attacker.get_adjacent_areas()
		for n in neigbours:
			if n.get_owner_name() != self.player_name or n.get_name() == self.attacker.get_name():
				continue
			else:
				self.possible_attacks.append(TreeNode(n, self.player_name))
		
		n_attacks = 0
		for subnode in self.possible_attacks:
			n_attacks += subnode.unpack_to_n_level(n - 1)
		return n_attacks
	
	def get_paths_to_n_level(self, n, paths=[], cur_path=[]):
		if n <= 0:
			paths.append(cur_path)
			return paths
		cur_path.append(self.attacker)
		
		loop_attacks = 0
		if len(self.possible_attacks) > 0:
			for subnode in self.possible_attacks:
				if subnode.attacker in cur_path:
					loop_attacks +=1
					continue
				paths = subnode.get_paths_to_n_level(n-1, paths, cur_path)
			if loop_attacks == len(self.possible_attacks):
				paths.append(cur_path)
			return paths
		else:
			paths.append(cur_path)
			return paths

