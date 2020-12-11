import copy
import numpy as np
from dicewars.ai.utils import probability_of_successful_attack


class UCS():
	"""
	Uniform cost search for searching best possible way to attack,
	simulating attacks and searching fields
	"""

	def __init__(self, player_name, KNN) -> None:
		self.player_name = player_name
		self.knn = KNN

	def get_enemies_of(self, area, board) -> list:
		""" Returns enemy neigbour areas of given area

		Parameters:
		-----------
			area (Area): Area to get enemies from

		Returns:
		--------
			list[Area]: List of enemy areas
		"""
		return [board.get_area(a) for a in area.get_adjacent_areas() if board.get_area(a).get_owner_name() != self.player_name]
		
	def get_regionsize_change_with(self, areas, board) -> int:
		""" calculate size-change of largest region of player after capturing gicen areas

		Parameters:
		-----------
			areas (Area | [Area]): Area to simulate as captured
			board (Board): Play board

		Returns:
		--------
			int: change of largest size of region
		"""
		old_size = self.get_biggest_regionsize(board)		
		boardcopy = copy.deepcopy(board)
		if type(areas) not in (list, tuple):
			areas = [areas]
		for area in areas:
			area = boardcopy.get_area(area.get_name())
			area.owner_name = self.player_name
		new_size = self.get_biggest_regionsize(boardcopy)
		return new_size - old_size

	def get_biggest_regionsize(self, board) -> int:
		""" Returns size of biggest region of player on given board

		Parameters:
		-----------
			board (Board): Play board

		Returns:
		--------
			int: size of largest region
		"""
		regions = board.get_players_regions(self.player_name)
		return max([len(r) for r in regions])

	def simulate_attacks(self, attacker, target, board):
		""" Simulate given attacks as succesfull and modify copy of a board

		Parameters:
		-----------
			attacks (list[Area, Area]): list of attaks [attacker, target]
			board (Board): board from which to play attacks

		Returns:
		--------
			Board: Returns modified copy of the given board
		"""

		attacker = board.get_area(attacker)
		target = board.get_area(target)

		dices = attacker.get_dice()
		if dices <= 1:
			return board

		attacker.dice = 1
		target.dice = dices - 1
		target.owner_name = self.player_name
	
		return board


	def create_datapoint(self, attacker, target, board):
		""" Creates datapoint for KNN evaluation

		Parameters:
		-----------
			attacker (Area): Attacking area
			target (Area): Target area
			board (Board): Playing board

		Returns:
		--------
			np.array(): Numpy array of features
		"""
		attacker = board.get_area(attacker)
		target = board.get_area(target)
		dp = []
		dp.append(probability_of_successful_attack(board, attacker.get_name(), target.get_name()))
		dp.append(self.get_regionsize_change_with(target, board))
		enemies = self.get_enemies_of(target, board)
		if len(enemies) == 0:
			dp.append(0)
		else:
			dp.append(np.array([x.get_dice() for x in enemies]).mean())
		
		enemies = self.get_enemies_of(attacker, board)
		dp.append(np.array([x.get_dice() for x in enemies]).mean())

		return np.asarray(dp)

	def evaluate_attack(self, dp, board):
		""" Evaluates single attack from attacker to target

		Parameters:
		-----------
			attacker (int): Attacking area name
			target (int): Target area name
			board (Board): Playing board

		Returns:
		--------
			float: result from KNN classifier
		"""
		# attacker = board.get_area(attacker)
		# target = board.get_area(target)
		return self.knn.evaluate(dp)

	def propagade_results(self, board, attacks):
		""" Is used for training purpuses
			Takes saved attacks from last turn and store them into dataset based on results of an attacks

		Args:
		-----
			board (Board): Game board
			attacks ([target_id, [datapoint, value_of_attack]]): Stored attacks
		"""

		if attacks != {}:
			for target, value in attacks.items():
				dp, index = value
				if index == 0.5 and board.get_area(target).get_owner_name() == self.player_name:
					index = 1

				# print("index = ", index)
				self.knn.set_new_datapoint(np.asarray(dp), index)

			self.knn.save_dataset()