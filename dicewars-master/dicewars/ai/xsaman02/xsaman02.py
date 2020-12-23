import logging, time, copy
import numpy as np

from dicewars.ai.utils import possible_attacks
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.xsaman02.UCS.TreeSearch import TreeSearch
from dicewars.ai.xsaman02.KNN.KNN import KNN
from dicewars.ai.xsaman02.UCS.UCS import UCS

class AI:
    """
    AI using Tree search for finding possible attack paths 
    and evaluating them using KNN klassifier
    """
    
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        # self.logger = logging.getLogger('AI')

        d = {"probability of capture" : [0, 1], 
		     "change of biggest region size after attack" : [0,8], 
		     "mean dice of enemy terrs. of target" : [0, 8], 
		     "mean dice of enemy terrs. of source" : [0, 8]}
        #Initialization of KNN classifier
        knn = KNN(11, list(d.values()), np.array([1, 1, 1, 1]))
        knn.load_dataset()
        #Inicialization of UCS algoritm
        self.ucs = UCS(player_name, knn)
        
        # {target_id : ((datapoint), value)} 
        # value is needed because of eventual 0.5 value when attack was succesfull
        self.last_turn_attacks = {}
        # [(datapoint), target_id]
        self.last_attack = []
        self.conquered_provinces = {}
        self.last_move = []
        self.time_start = 0

        self.tmp_mean_time = []

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):

        # TO CHANGE - learning rate
        lr = 0.03

        # If it's first move this turn -> Evaluate new strategy & propagade_results  from last turn
        if nb_moves_this_turn == 0:
            self.time_start = time.perf_counter()
            self.ucs.propagade_results(board, self.last_turn_attacks)
            self.evaluate_strategy(board, nb_turns_this_game, time_left)
        elif self.last_attack != []:
            dp_val = 0.5 if board.get_area(self.last_attack[1]).get_owner_name() == self.player_name else 0
            self.last_turn_attacks[self.last_attack[1]] = (self.last_attack[0], dp_val)
            self.last_attack = []

        # parsing all sorted attacks and executing them them
        for attack_path, index in self.evaluated_attacks:
            for _ in range(len(attack_path)-1):
                attacker = board.get_area(attack_path[0])
                target = board.get_area(attack_path[1])
                # Evaluating if attack is possible
                if attacker.get_owner_name() == self.player_name and target.get_owner_name() != self.player_name and attacker.can_attack():
                    # TODO Comment out
                    # # This if is used for learning process. 
                    # if np.random.random(1) < lr:
                    #     self.last_attack = [self.ucs.create_datapoint(attack_path[0], attack_path[1], board), attack_path[1]]
                    
                    if self.last_move != [] and board.get_area(self.last_move[1]).get_owner_name() == self.player_name:
                        self.conquered_provinces[self.last_move[1]] = self.last_move[0]
                    
                    self.last_move = attack_path[:2]
                    #Poping attacker from path
                    #Example -> [12, 23, 2] -> [23, 2]
                    attack_path.pop(0)
                    # If there is no target left -> Remove attack path from list of sorted attacks
                    if len(attack_path) == 1:
                        self.evaluated_attacks.remove([attack_path, index])
                    return BattleCommand(attacker.get_name(), target.get_name())
                else:
                    # If attack from last move was succesfull -> save it
                    if self.last_move != [] and board.get_area(self.last_move[1]).get_owner_name() == self.player_name:
                        self.conquered_provinces[self.last_move[1]] = self.last_move[0]
                    self.last_move = []

                    # conquerer of province is same as attacker
                    if attack_path[1] in self.conquered_provinces and attack_path[0] == self.conquered_provinces[attack_path[1]]:
                        attack_path.pop(0)
                        if len(attack_path) == 1:
                            self.evaluated_attacks.remove([attack_path, index])
                            break
                    else:
                        # if conquerer != attacker -> remove attack path from list
                        self.evaluated_attacks.remove([attack_path, index])
                        break

        
        # print("{}. turn took {:.3f}s, with time left {}".format(nb_turns_this_game, time.perf_counter() - self.time_start, time_left),end="\n")
        self.last_move = []
        self.conquered_provinces = {}
        self.tmp_mean_time.append(time.perf_counter() - self.time_start)

        return EndTurnCommand()


    def evaluate_strategy(self, board, nb_turns_this_game, time_left, threshold_for_attacking=0.5):
        """ Evaluates best possible paths for attacking 
            according to classifier.

        Args:
        -----
            - board (board): Game board
            - `nb_turns_this_game` (int): number of turn this game
            - `time_left` (float): time left
            - `threshold_for_attacking` (float, optional): Threshold for cancel evaluating path, when classified value is below. Defaults to 0.5.
        """
        self.evaluated_attacks = {}
        evaluated_datapoints = {}
        tree = TreeSearch(self.player_name)
        restrict = self.get_depth_restriction(time_left)
        start = time.perf_counter()

        cached = 0
        uncached = 0

        attacks = possible_attacks(board, self.player_name)
        attackers = list(set([source for source, _ in attacks]))
        # For all provinces which player can attack
        for source in attackers:
            # Evaluate all paths for given province
            paths = tree.get_paths(source, restrict, board)

            for path in paths:
                path_index = 0
                board_copy = copy.deepcopy(board)

                # sequentially evaluates given path
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
                        # sequentially stores path prefixes with it's indexis
                        path_index += res
                        self.evaluated_attacks[tuple(path[:i+2])] = path_index / (i+1)
                        self.ucs.simulate_attacks(path[i], path[i+1], board_copy)
                    else:
                        # If result of partial attack is below threshold -> stop evaluating given path
                        break
                
                # print("{}. path evaluated in {}s".format(p+1, time.perf_counter() - start))
        
        self.evaluated_attacks = sorted(self.evaluated_attacks.items(), key=lambda x: x[1], reverse=True)
        self.evaluated_attacks = [[list(path), index] for path, index in self.evaluated_attacks]
        # print("\n{}. turn evaluated strategy in {}s".format(nb_turns_this_game, time.perf_counter() - start))
        # print(f'evaluations: cached = {cached}, uncached = {uncached}')


    def get_depth_restriction(self, time_left):
        if time_left < 0.5:
            return 2
        elif time_left < 1:
            return 3
        elif time_left < 2:
            return 4
        elif time_left < 5:
            return 5
        elif time_left < 7:
            return 6
        else:
            return 7
    