import logging, time, copy
import numpy as np

from dicewars.ai.utils import possible_attacks
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.sui_ai.TreeSearch import TreeSearch
from dicewars.ai.sui_ai.KNN.KNN import KNN
from dicewars.ai.sui_ai.UCS.UCS import UCS


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        d = {"probability of capture" : [0, 1], 
		     "change of biggest region size after attack" : [0,15], 
		     "mean dice of enemy terrs. of target" : [0, 8], 
		     "mean dice of enemy terrs. of source" : [1, 8]}
        knn = KNN(11, list(d.values()), np.array([1, 1.2, 1.3, 1.3]))
        # knn.initialize(10, len(d.keys()))
        # knn.save_dataset()
        knn.load_dataset()
        self.ucs = UCS(player_name, knn)
        
        # {target_id : ((datapoint), value)} 
        # value is needed because of eventual 0.5 value when attack wa succesfull
        self.last_turn_attacks = {}
        # [(datapoint), target_id]
        self.last_attack = []
        self.time_start = 0

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        lr = 0.1
        if nb_moves_this_turn == 0:
            self.time_start = time.perf_counter()
            self.ucs.propagade_results(board, self.last_turn_attacks)
            self.evaluate_strategy(board, nb_turns_this_game, time_left, 0.4)
        elif self.last_attack != []:
            dp_val = 0.5 if board.get_area(self.last_attack[1]).get_owner_name() == self.player_name else 0
            self.last_turn_attacks[self.last_attack[1]] = (self.last_attack[0], dp_val)
            self.last_attack = []

        for attack_path, index in self.evaluated_attacks:
            # print(attack_path)
            for _ in range(len(attack_path)-1):
                attacker = board.get_area(attack_path[0])
                target = board.get_area(attack_path[1])
                if attacker.get_owner_name() == self.player_name and target.get_owner_name() != self.player_name and attacker.can_attack():
                    # This if is used for learning procces. 
                    # TODO Comment out
                    # if np.random.random(1) < lr:
                    #     self.last_attack = [self.ucs.create_datapoint(attack_path[0], attack_path[1], board), attack_path[1]]
                    attack_path.pop(0)
                    if len(attack_path) == 1:
                        self.evaluated_attacks.remove([attack_path, index])
                    return BattleCommand(attacker.get_name(), target.get_name())
                else:
                    # print("attack already proceeded for ", attack_path)
                    self.evaluated_attacks.remove([attack_path, index])
                    break
            
        print("{}. turn took {:.3f}s".format(nb_turns_this_game, time.perf_counter() - self.time_start),end="\n")
        # print("number of paths to attack: {}".format(len(self.evaluated_attacks)))
        return EndTurnCommand()


    def evaluate_strategy(self, board, nb_turns_this_game, time_left, threshold_for_attacking=0.5):
        self.evaluated_attacks = {}
        evaluated_datapoints = {}
        tree = TreeSearch(self.player_name)
        restrict = self.get_depth_restriction(time_left)
        start = time.perf_counter()

        cached = 0
        uncached = 0

        attacks = possible_attacks(board, self.player_name)
        attackers = list(set([source for source, _ in attacks]))

        for source in attackers:
            paths = tree.get_paths(source, restrict, board)
            for p, path in enumerate(paths):
                path_index = 0
                board_copy = copy.deepcopy(board)
                for i in range(len(path)-1):
                    dp = self.ucs.create_datapoint(path[i], path[i+1], board_copy)
                    if tuple(dp) in evaluated_datapoints:
                        res = evaluated_datapoints[tuple(dp)]
                        cached += 1
                    else:
                        res = self.ucs.evaluate_attack(copy.deepcopy(dp), board)
                        evaluated_datapoints[tuple(dp)] = res
                        uncached += 1
                    
                    if res > threshold_for_attacking:
                        path_index += res
                        self.evaluated_attacks[tuple(path[:i+2])] = path_index / (i+1)
                        self.ucs.simulate_attacks(path[i], path[i+1], board_copy)
                    else:
                        break
                
                # print("{}. path evaluated in {}s".format(p+1, time.perf_counter() - start))
        
        self.evaluated_attacks = sorted(self.evaluated_attacks.items(), key=lambda x: x[1], reverse=True)
        self.evaluated_attacks = [[list(path), index] for path, index in self.evaluated_attacks]
        print("\n{}. turn evaluated strategy in {}s".format(nb_turns_this_game, time.perf_counter() - start))
        # print(f'evaluations: cached = {cached}, uncached = {uncached}')


    def get_depth_restriction(self, time_left):
        if time_left < 1:
            return 3
        elif time_left < 2:
            return 4
        elif time_left < 5:
            return 5
        elif time_left < 7:
            return 6
        else:
            return 7
    