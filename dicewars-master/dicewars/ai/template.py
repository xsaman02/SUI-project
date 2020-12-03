from dicewars.ai.sui_ai.TreeSearch import TreeSearch
# from dicewars.ai.sui_ai.KNN.KNN import KNN
import logging, time, copy
import numpy as np

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area

from dicewars.ai.sui_ai.KNN.KNN import KNN
from dicewars.ai.sui_ai.UCS.UCS import UCS
from numpy import random

class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        d = {"probability of capture" : [0, 1], 
		     "change of biggest region size after attack" : [0,15], 
		     "mean dice of enemy terrs. of target" : [0, 8], 
		     "attacker dice" : [1, 8]}
        knn = KNN(11, list(d.values()), np.array([1, 1.2, 1.4, 1.5]))
        knn.initialize(100, len(d.keys()))
        knn.save_dataset()

        self.ucs = UCS(player_name, knn)


    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        threshold_for_attacking = 0.5
        evaluated_attacks = {}
        tree = TreeSearch(self.player_name)
        start = time.perf_counter()


        attacks = possible_attacks(board, self.player_name)
        attackers = list(set([source for source, _ in attacks]))

        for source in attackers:
            paths = tree.get_paths(source, board)

            for path in paths:
                path_index = 0
                board_copy = copy.deepcopy(board)
                for i in range(len(path)-1):
                    res = self.ucs.evaluate_attack(path[i], path[i+1], board)
                    if res > threshold_for_attacking:
                        path_index += res
                        evaluated_attacks[tuple(path[i:i+2])] = path_index / i+1
                        self.ucs.simulate_attacks(path[i], path[i+1], board_copy)
                    else:
                        break
        
        evaluated_attacks = sorted(evaluated_attacks.items(), key=lambda x: x[1], reverse=True)
        print("{} turn. Evaluated in {}".format(nb_turns_this_game, time.perf_counter() - start))

        return EndTurnCommand()
        




        province_attacks = tree.evaluate_possible_paths(board, self.player_name, 10)
        # print(province_attacks.items())

        print("got paths in", time.perf_counter() - start)
        #sorted dict descending
        province_attacks = sorted(province_attacks.items(), key=lambda x: x[1], reverse=True)
        print("sorted paths in", time.perf_counter() - start)

        print(f'length of attacks = {len(province_attacks)}')
        i_tmp = 0

        # for every province that is able to attack someone (provinces sorted descending)
        for province, n_attacks in province_attacks:
            i_tmp += 1
            print(f'i = {i_tmp}')
            paths = tree.get_paths_of(province, board.get_area(province).get_dice())
            print(type(paths))
            print(paths.shape)
            print(paths)

            paths = np.unique(paths, axis=0)
            print("passed unique()")

            # for every possible path of attack from single province
            for path in paths:
                boardcopy = copy.deepcopy(board)
                last_result = 0

                # indexing throught the path of single attack
                for i in range(len(path) - 1):
                    """ takes cumulatively whole attack path and evaluates each attack
                        for each attack which is good enough is created "simulation" and board is modified
                        results are calculated as aritmethic mean and stored in dictionary
                    """
                    result = self.ucs.evaluate_attack(path[i], path[i+1], board)
                    if result > threshold_for_attacking:
                        last_result += result
                        boardcopy = self.ucs.simulate_attacks([[path[i], path[i+1]]], boardcopy)
                        evaluated_attacks[tuple([x for x in path[:i+2]])] = last_result / (i + 1)
                    else:
                        # In case of no good attack 
                        # current attack path is stored and rest of path is not evaluated 
                        break

        print("evaluated paths in", time.perf_counter() - start)
        del tree
        return EndTurnCommand()


        evaluated_attacks = sorted(evaluated_attacks.items(), key=lambda x: x[1], reverse=True)
        print("sorted best attack paths in", time.perf_counter() - start)
        for path, result in evaluated_attacks:
            print(path, result)
        

        return EndTurnCommand()