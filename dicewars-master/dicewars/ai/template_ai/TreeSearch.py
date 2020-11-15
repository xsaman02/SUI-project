import numpy as np
from dicewars.ai.template_ai.TreeNode import TreeNode

class TreeSearch():
	"""
	Class used for finding best way to attack the enemy provinces
	"""
	
	board = None
	node = None

	def __init__(self, board) -> None:
		"""Init function

		Parameters:
		-----------
			board (dicewars.client.game.Board):
		"""
		self.board = board

	def create_structure(self) ->None
	