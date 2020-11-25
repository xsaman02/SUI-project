from dicewars.ai.sui_ai.KNN.KNN import KNN
import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area

from dicewars.ai.sui_ai.UCS.UCS import UCS
from numpy import random

class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        d = {"probability of capture" : [0, 1], 
		     "change of biggest region size after attack" : [0,15], 
		     "mean dice of enemy terrs. of target" : [1, 8], 
		     "attacker dice" : [1, 8]}
        knn = KNN(11, list(d.values()))
        knn.initialize(100, len(d.keys()))
        # knn.load_dataset()

        self.ucs = UCS(player_name, knn)
        self.simulated_attacks = []
        self.last_turn_attacks = []
        print("init completed")

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        if self.simulated_attacks == []:

            if time_left >= 6:
                recursion_depth = 3
            elif time_left >= 2:
                recursion_depth = 2
            else:
                recursion_depth = 1
            
            self.ucs.propagade_results(board, self.last_turn_attacks)
            print("propagaded")
            self.evaluate_strategy(board, recursion_depth)
            print("evaluated best turns")
        
        for turn in self.simulated_attacks:
            for attacker, target in turn:
                print("made an attack")
                return BattleCommand(attacker.get_name(), target.get_name())


        self.simulated_attacks = []
        self.last_turn_attacks = []
        return EndTurnCommand()


    def evaluate_strategy(self, board, max_recursion_depth, recursion_depth=0):
        print("recursion depth: ",recursion_depth)
        if recursion_depth >= max_recursion_depth:
            print("recursion depth exceeded. Returning.")
            return

        attacks = list(possible_attacks(board, self.player_name))
        print("listed attacks")

        attacked_targets = []
        used_attackers = []
        attacks_to_simulate = []
        for _, target in attacks:
            if target in attacked_targets:
                continue
            
            attacked_targets.append(target)
            print("evaluating all possible attacks on target")
            results = self.ucs.evaluate_all_attacks_on(target, board, attacks)
            print("Choosing best attack", results)
            best_choise = self.get_best_attack(results, used_attackers)
            if type(best_choise) == type(None):
                continue

            print("adding attacker & attack")
            used_attackers.append(best_choise[0])
            attacks_to_simulate.append([best_choise[0], target])

            print("choosing to create training data")
            # creates learning process where some of attacks will propagate next turn into dataset
            if random.rand() > 0.8:
                self.last_turn_attacks.append(best_choise[2], target)
                print("choosing data added")

        print("simulating attacks")
        boardcopy = self.ucs.simulate_attacks(attacks_to_simulate, board)
        print("adding simulated attacks")
        self.simulated_attacks.append(attacks_to_simulate)
        print("raising recursion level")
        return self.evaluate_strategy(boardcopy, max_recursion_depth, recursion_depth + 1)

    def get_best_attack(self, results, used_attackers):

        best_val = 0
        best_index = None
        for i, result in enumerate(results):
            # constant in if statement is minimum value for attack index form KNN
            if result[1] > best_val and result[1] > 0.1 and result[0] not in used_attackers:
                best_val = result[1]
                best_index = i
        
        return results[best_index] if best_index != None else None