import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack

from dicewars.ai.template_ai.KNN import KNN


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        # print(len(board.areas))


    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        attacks = list(possible_attacks(board, self.player_name))

        for source, target in attacks:
            prob = probability_of_successful_attack(board, source.get_name(), target.get_name())
            if prob > 0.6:
                return BattleCommand(source.get_name(), target.get_name())

        return EndTurnCommand()
