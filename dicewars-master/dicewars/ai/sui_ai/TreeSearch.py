from typing import Iterator
import numpy as np
from dicewars.ai.utils import possible_attacks
from dicewars.ai.sui_ai.TreeNode import TreeNode

class TreeSearch():
	"""
	Class used for finding best way to attack the enemy provinces
	"""
	def __init__(self) -> None:
		"""Init function
		"""
		self.paths = None


	def evaluate_possible_paths(self, board, player_name, immersion) -> list(int):
		"""Evaluate possible attacking paths to n level from all provinces and returns list of integers
		   of possible attacks from different provinces

		Parameters:
		-----------
			board (Board): Board of the game
			player_name (int): Player's name
			immersion (int): recursion number to expand in search tree

		Returns:
		--------
			list(int) : number of attacks from different provinces (all possible ways)
		"""

		attacks = possible_attacks(board, player_name)
		attackers = list(set([x[0] for x in attacks]))
		self.paths = [TreeNode(x, player_name) for x in attackers]
		n_attacks = []
		for attacker in self.paths:
			n_attacks.append(attacker.expand_to_n_level(immersion))
		return n_attacks


	def get_paths_to_n_level(self, n) -> list():
		""" Using generated tree structure from function evaluate_possible_paths()
		    Function returns 3D-tuple which represents:
			1D - list of paths sorted by starting province
			2D - list of paths from one province
			3D - list of individual provinces to attack (from own province to last enemy province)

			Arrays doens't need to be same size as some attacks can be possibly shorter

		Parameter:
		----------
			n (int): recursion depth to search tree (if tree is not big enough, returns longest way possible)

		Returns:
		--------
			3D array
		"""
		assert(type(self.paths) != type(None))

		provinces_paths = []
		for path in self.paths:
			provinces_paths.append(path.get_paths_to_n_level(n))			

		return provinces_paths

