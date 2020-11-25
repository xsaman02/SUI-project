import copy
import numpy as np
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack


class UCS():
	"""
	Uniform cost search for searching best possible way to attack,
	simulating attacks and searching fields
	"""

	boardcopy = None
	max_dataset_length = 500

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

	def simulate_attacks(self, attacks, board):
		""" Simulate given attacks as succesfull and modify copy of a board

		Parameters:
		-----------
			attacks (list[Area, Area]): list of attaks [attacker, target]
			board (Board): board from which to play attacks

		Returns:
		--------
			Board: Returns modified copy of the given board
		"""
		if self.boardcopy == None:
			self.boardcopy = copy.deepcopy(board)

		for attacker, target in attacks:
			attacker = self.boardcopy.get_area(attacker)
			target = self.boardcopy.get_area(target)

			dices = attacker.get_dice()
			if dices == 1:
				continue

			attacker.dice = 1
			target.dice = dices - 1
			target.owner_name = self.player_name
		
		return self.boardcopy


	def set_boardcopy(self, board):
		""" Setup copy of the board

		Parameters:
		-----------
			board (Board): board to copy
		"""
		self.boardcopy = copy.deepcopy(board)

	def clear_boardcopy(self):
		""" removes modified board
		"""
		self.boardcopy = None

	def get_boardcopy(self):
		"""Getter for modified board

		Returns:
		--------
			Board: saved board
		"""
		return self.boardcopy

	def evaluate_all_attacks_on(self, target, board, attacks):
		"""Evaluate all possible attacks on target from board's areas
			Allows you to enter modified board using 'simulate-attacks' function
			and simulate multi-level attacks.

		Parameters:
		-----------
			target (Area): Target area to attack from
			board (Board): Playing [modified] board
			attacks (Iterator[ [Area, Area] ]): possible attacks

		Returns:
		--------
			2D list: list where each element is: (attacker-area, evaluation, datapoint)
		"""
		print("starting to evaluate")
		results = []
		for a, t in attacks:
			if t == target and a.get_dice() > 1:
				print("creating datapoint")
				dp = self.__create_datapoint(a, t, board)
				print("evaluating")
				print(dp,type(dp))
				results.append([a, self.knn.evaluate(dp), dp])
				print("evaluated: ", results[-1][1])

		return results

	def __create_datapoint(self, attacker, target, board):
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
		dp = []
		dp.append(probability_of_successful_attack(board, attacker.get_name(), target.get_name()))
		print("prob. added")
		dp.append(self.get_regionsize_change_with(target, board))
		print("region size added")
		dp.append(np.array([x.get_dice() for x in self.get_enemies_of(target, board)]).mean())
		print("mean dice added")
		dp.append(attacker.get_dice())
		print("dice added")

		return np.asarray(dp)

	def evaluate_attack(self, attacker, target, board):
		""" Evaluates single attack from attacker to target

		Parameters:
		-----------
			attacker (Area): Attacking area
			target (Area): Target area
			board (Board): Playing board

		Returns:
		--------
			float: result from KNN classifier
		"""
		dp = self.__create_datapoint(attacker, target, board)

		return self.knn.evaluate(dp)

	def propagade_results(self, board, attacks):
		if self.knn.get_len_of_dataset() >= self.max_dataset_length:
			self.knn.create_new_dataset()

		for dp, target in attacks:
			self.knn.set_new_datapoint(np.asarray(dp), board.get_area(target.get_name()).get_owner_name() == self.player_name)

		self.knn.save_dataset()