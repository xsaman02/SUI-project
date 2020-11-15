import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack



class AI:
    def __init__(self, player_name, board, players_order):
        print("hello")
        self.player_name = player_name
        self.logger = logging.getLogger('AI')


    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        attacks = list(possible_attacks(board, self.player_name))

        for source, target in attacks:
            print(type(source))
            break
            # prob = probability_of_successful_attack(board, source.get_name(), target.get_name())
            # if prob > 0.6:
            #     return BattleCommand(source.get_name(), target.get_name())

        return EndTurnCommand()
