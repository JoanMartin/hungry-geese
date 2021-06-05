from random import choice

from kaggle_environments.envs.hungry_geese.hungry_geese import translate, Action

from game_state import GameState


class RandomGoose:

    @staticmethod
    def select_moves(game_state: GameState):
        all_actions = [action for action in Action]
        moves = []
        final_action = None

        for goose in game_state.geese:
            candidates = []
            head_adjacent_candidates = []

            if goose.status == "ACTIVE":
                for action in all_actions:
                    new_position = translate(goose.positions[0], action, game_state.columns, game_state.rows)
                    if not goose.is_last_action(action) and game_state.is_valid_position(new_position):
                        if game_state.is_head_adjacent_position(goose.index, new_position):
                            head_adjacent_candidates.append(action)
                        else:
                            candidates.append(action)

                if len(candidates) > 1:
                    final_action = choice(candidates)
                elif any(head_adjacent_candidates) or any(candidates):
                    final_action = choice(head_adjacent_candidates + candidates)
                else:
                    final_action = choice(all_actions)

            moves.append(final_action)

        return moves
