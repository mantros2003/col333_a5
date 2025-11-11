import time
import random
import math

class Game:
    ...

class GameNode:
    def __init__(self, game, par = None, state = None, action_taken = None, player: int = 1) -> None:
        self.game = game
        self.par = par
        self.action_taken = action_taken
        self.state = state if state else self.game.get_start_state()
        self.children: dict = {}  # Stores actions - GameNode mapping
        self.player = player

        self.visit_cnt = 0
        self.value = 0

        self.unexpanded_actions = []

    def is_root(self) -> bool:
        return self.par is None
    
    def is_terminal(self) -> bool:
        return (self.state.get("move_num", 0) == 500) or self.game.is_over()
    
    def is_fully_expanded(self) -> bool:
        return True
    
    def value_estimate(self) -> float:
        return self.value / self.visit_cnt if self.visit_cnt else 0

class GameTree:
    def __init__(self, game, root: GameNode) -> None:
        self.game = game
        self.root: GameNode = root
        self.states: set = set()
        # Add the root node to states table

class MCTS:
    def __init__(self, game, rollout_limit, c_explore: float = 1.414):
        self.game = game
        self.rollout_limit = rollout_limit
        self.c_explore = c_explore
        # Initialize an empty game tree
        self.game_tree = GameTree(self.game, GameNode(self.game))
    
    def search(self, max_simulations = 1000, time_limit = None):
        start = time.perf_counter()
        num_sims = 0

        while (num_sims < max_simulations) and (time_limit is None or (time.perf_counter() - start < time_limit)):
            self._simulate(self.game_tree.root)
            num_sims += 1
    
    def best_action(self, node: GameNode, temp: float = 0.0):
        child_visits = [(action, child.visit_cnt) for action, child in node.children.items()]

        if temp <= 0.0:
            return max(child_visits, key = lambda x: x[1])[1]
        
        visits = [item[1] for item in child_visits]
        actions = [item[0] for item in child_visits]
        
        powered_visits = [v**(1.0 / temp) for v in visits]
        total_visits = sum(powered_visits)
        
        if total_visits == 0:
            return random.choice(actions)

        # Create a probability distribution
        probs = [v / total_visits for v in powered_visits]
        
        # Sample one action based on the distribution
        return random.choices(actions, weights=probs, k=1)[0]
    
    def _simulate(self, node: GameNode):
        curr = node
        while not curr.is_terminal():
            if not curr.is_fully_expanded():
                new_child = self._expand(curr)
                reward = self._rollout(new_child)
                self._backprop(new_child, reward)
            else:
                curr = self._select_best_child(node)
        
        reward = self.game.get_reward(curr.state, curr.par.player if curr.par else None)
    
    def _select_best_child(self, node: GameNode) -> GameNode:
        """
        Selects the best child node based on the UCB1 (UCT) formula.
        This is for a negamax framework.
        """
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            if child.visit_cnt == 0:
                return child

            exploit_score = -child.value / child.visit_cnt

            explore_score = self.c_explore * math.sqrt(
                math.log(node.visit_cnt) / child.visit_cnt
            )

            score = exploit_score + explore_score

            if score > best_score:
                best_score = score
                best_child = child

        return best_child


    def _rollout(self, node: GameNode):
        """
        Performs a random simulation (rollout) from the given node.
        Returns the reward from the perspective of the *node's player*.
        """
        curr_state = node.state

        while not self.game.is_terminal(curr_state):
            actions = self.game.get_actions(curr_state)

            random_action = random.choice(list(actions))

            curr_state = self.game.transition(curr_state, random_action)

        return self.game.get_reward(curr_state, node.player)
    
    def _expand(self, node: GameNode) -> GameNode:
        """
        Expands one unvisited action from the given node.
        """
        action = node.unexpanded_actions.pop()

        next_state = self.game.transition(node.state, action)

        new_child = GameNode(
            self.game,
            par=node,
            state=next_state,
            action_taken=action
        )

        node.children[action] = new_child

        return new_child

    def _backprop(self, node: GameNode, reward):
        """
        Updates the value and visit counts of nodes up the tree (negamax-style).
        """
        curr = node
        # The reward is from the perspective of `node.player`.
        # We must flip the reward for each alternating player up the tree. 
        while curr is not None:
            curr.visit_cnt += 1
            curr.value += reward
            
            # Flip the reward for the parent node
            reward = -reward
            
            curr = curr.par