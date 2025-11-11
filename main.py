from torch import nn

from starter.client_server.student_agent import StudentAgent

class GameNode:
    def __init__(self, game, par = None, state = None) -> None:
        self.game = game
        self.par = par
        self.state = state if state else self.game.get_start_state()

        self.visit_cnt = 0
        self.value = 0

    def is_root(self) -> bool:
        return self.par is None
    
    def value_estimate(self) -> float:
        return self.value / self.visit_cnt if self.visit_cnt else 0

class GameTree:
    def __init__(self, game, root: GameNode) -> None:
        self.game = game
        self.root: GameNode = root
        self.states: dict = dict()

class MCTS:
    def __init__(self, game, rollout_limit):
        self.game = game
        self.rollout_limit = rollout_limit
        # Initialize an empty game tree
        self.game_tree = GameTree(self.game, GameNode(self.game))
    
    def _rollout(self):
        curr_state = self.game.get_state()
        actions = self.game.get_actions(curr_state)
        next_states = {a: self.game.transition(a) for a in actions}