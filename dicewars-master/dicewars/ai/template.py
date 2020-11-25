from dicewars.ai.sui_ai.TreeSearch import TreeSearch
# from dicewars.ai.sui_ai.KNN.KNN import KNN
import logging
import time

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area

# from dicewars.ai.sui_ai.UCS.UCS import UCS
# from dicewars.ai.sui_ai.UCS.Attack_struct import Attack
# from numpy import random

class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        self.tree = TreeSearch()
        res = self.tree.evaluate_possible_paths(board, self.player_name, 10)
        # d = {"probability of capture" : [0, 1], 
		#      "change of biggest region size after attack" : [0,15], 
		#      "mean dice of enemy terrs. of target" : [1, 8], 
		#      "attacker dice" : [1, 8]}
        # knn = KNN(11, list(d.values()))
        # knn.initialize(100, len(d.keys()))
        # knn.load_dataset()

        # self.ucs = UCS(player_name, knn)
        # self.simulated_attacks = []
        # self.last_turn_attacks = []
        # print("init completed")

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        start = time.perf_counter()
        stop = time.perf_counter()
        print(res)
        print(stop-start)

        res = tree.get_paths_of(0, 4)
        print(res.size)



        return EndTurnCommand()

        # if time_left < 3:
        #    max_evaluates = 50
        # elif time_left < 5:
        #     max_evaluates = 100
        # elif time_left < 7:
        #     max_evaluates = 150
        # elif time_left < 9:
        #     max_evaluates = 200
        # else:
        #     max_evaluates = 300

        # boarders = board.get_player_border(self.player_name)
        # n_evaluates = 0
        # for area in boarders:
        #     self.ucs.evaluate_all_paths_from(area, board)