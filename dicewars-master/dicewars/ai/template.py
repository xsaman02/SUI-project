import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area

from dicewars.ai.sui_ai.KNN import KNN
import numpy as np

class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')

        self.knn = KNN(7)
        d = {"number of enemy terrs. of attacker" : [1, 10], "number of enemy terrs. of target" : [0, 10], "probability of capture" : [0, 1], "probability of sustain" : [0,1]}
        self.knn.initialize(d, 200)
        self.knn.set_deviation(0.1)
        pos = np.array([1, 1, 0.75, 0.75])
        neg = np.array([9, 8, 0.1, 0.2])
        print("starts mapping")
        self.knn.first_mapping(pos, neg)
        print("init completed")


    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        
        attacks = list(possible_attacks(board, self.player_name))

        for source, target in attacks:
            v = []
            v.append(get_n_enemy_terr(source, self.player_name))
            v.append(get_n_enemy_terr(target, self.player_name))
            v.append(probability_of_successful_attack(board, source.get_name(), target.get_name()))
            v.append(probability_of_holding_area(board, target.get_name(), source.get_dice() - 1, self.player_name))
            r = self.knn.evaluate(np.array(v))
            if r > 0.6:
                print("command")
                return BattleCommand(source.get_name(), target.get_name())

        return EndTurnCommand()

def get_n_enemy_terr(area, players_name):
    areas = area.get_adjacent_areas()
    return len([x for x in areas if x.get_owner_name() != players_name])