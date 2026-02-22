"""
mcts.py — Monte Carlo Tree Search with neural network guidance.

How MCTS works (the four phases, repeated N times per move):

  1. SELECT   — start at the root, repeatedly pick the child with the
                highest PUCT score until we reach an unexpanded leaf.

  2. EXPAND   — call the neural network on the leaf's board state.
                Create child nodes for every legal move, using the
                network's policy output as prior probabilities P(s, a).

  3. EVALUATE — use the neural network's *value* output as an estimate
                of who is winning (instead of rolling out to the end).

  4. BACKUP   — walk back up to the root, adding the value to every
                node's total and incrementing visit counts.
                The sign flips at each ply because it's always the
                *current player's* value that matters.

After N simulations the move distribution π is proportional to
visit counts raised to the power 1/temperature.

PUCT score (Predictor + UCT):
    PUCT(s, a) = Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))

  Q  = mean value over all simulations through this edge
  P  = neural network prior for action a in state s
  N  = total visits to parent node
  N(s,a) = visits to this child
  c_puct = exploration constant (typically 1.0 – 2.0)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import chess
import numpy as np

from env.chess_env import ChessEnv


@dataclass
class MCTSNode:
    """One node in the search tree — represents a board position."""

    prior:      float                         # P(s, a) from neural network
    board:      chess.Board = field(repr=False)
    parent:     Optional["MCTSNode"] = field(default=None, repr=False)
    move:       Optional[chess.Move] = None   # move that led HERE

    visit_count:  int   = 0
    total_value:  float = 0.0
    children:     Dict[chess.Move, "MCTSNode"] = field(default_factory=dict)
    is_expanded:  bool  = False

    @property
    def Q(self) -> float:
        """Mean action value (exploitation term)."""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def puct_score(self, c_puct: float) -> float:
        """PUCT = Q + U (exploration bonus)."""
        parent_n = self.parent.visit_count if self.parent else 1
        U = c_puct * self.prior * math.sqrt(parent_n) / (1 + self.visit_count)
        return self.Q + U

    def best_child(self, c_puct: float) -> "MCTSNode":
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct))


class MCTS:
    """
    Neural-network-guided Monte Carlo Tree Search.

    Usage
    -----
        mcts = MCTS(network, config)
        policy, moves = mcts.search(board)  # → probability distribution
        move = mcts.select_move(board)       # → chosen chess.Move
    """

    def __init__(self, network, config):
        self.network = network
        self.config  = config

    # ── Public interface ───────────────────────────────────────────────────────

    def search(
        self,
        board:     chess.Board,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, List[chess.Move]]:
        """
        Run MCTS from *board* for config.num_simulations steps.

        Returns
        -------
        policy : visit-count distribution over legal moves (sums to 1)
        moves  : the corresponding legal move list
        """
        root = MCTSNode(prior=1.0, board=board.copy())
        self._expand(root)

        # Dirichlet noise at root promotes exploration of rarely-visited moves
        if add_noise and root.children:
            moves  = list(root.children.keys())
            noise  = np.random.dirichlet([self.config.dirichlet_alpha] * len(moves))
            eps    = self.config.dirichlet_epsilon
            for m, n in zip(moves, noise):
                c = root.children[m]
                c.prior = (1 - eps) * c.prior + eps * n

        for _ in range(self.config.num_simulations):
            node      = root
            sim_board = board.copy()

            # 1. SELECT
            while node.is_expanded and node.children:
                node = node.best_child(self.config.c_puct)
                sim_board.push(node.move)

            # 2+3. EXPAND & EVALUATE  (or score terminal node)
            if sim_board.is_game_over(claim_draw=False):
                outcome = sim_board.outcome()
                if outcome.winner is None:
                    value = 0.0
                else:
                    # sim_board.turn is who moves NEXT; the last mover won/lost
                    value = -1.0 if outcome.winner == sim_board.turn else 1.0
            else:
                env = ChessEnv()
                env.board = sim_board
                state = env.encode_state()
                self._expand(node, env=env, state=state)

                mask         = self.network.get_legal_mask(env)
                _, value     = self.network.predict(state, mask, self.config.device)

            # 4. BACKUP
            self._backup(node, value)

        # Build policy from visit counts
        legal_moves = list(board.legal_moves)
        visits = np.array(
            [root.children[m].visit_count if m in root.children else 0
             for m in legal_moves],
            dtype=np.float32,
        )
        if visits.sum() > 0:
            policy = visits / visits.sum()
        else:
            policy = np.ones(len(legal_moves), dtype=np.float32) / max(len(legal_moves), 1)

        return policy, legal_moves

    def select_move(
        self,
        board:     chess.Board,
        temperature: float = 1.0,
        add_noise:   bool  = True,
    ) -> Optional[chess.Move]:
        """
        Run MCTS and return a move sampled according to *temperature*.

        temperature=0  → always pick the most-visited move (greedy / exploit)
        temperature=1  → sample proportionally to visit counts  (explore)
        """
        policy, moves = self.search(board, add_noise=add_noise)
        if not moves:
            return None

        if temperature < 1e-6:
            return moves[int(np.argmax(policy))]

        probs  = policy ** (1.0 / temperature)
        probs /= probs.sum()
        return np.random.choice(moves, p=probs)  # type: ignore[arg-type]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _expand(
        self,
        node:  MCTSNode,
        env:   Optional[ChessEnv] = None,
        state: Optional[np.ndarray] = None,
    ):
        """Create child nodes using network policy as priors."""
        if env is None:
            env = ChessEnv()
            env.board = node.board
        if state is None:
            state = env.encode_state()

        mask   = self.network.get_legal_mask(env)
        policy, _ = self.network.predict(state, mask, self.config.device)

        for move, action in env.legal_moves_with_actions():
            prior = float(policy[action]) if action < len(policy) else 1e-8
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(
                prior=max(prior, 1e-8),
                board=child_board,
                parent=node,
                move=move,
            )

        node.is_expanded = True

    def _backup(self, node: MCTSNode, value: float):
        """Propagate value to root, flipping sign at each ply."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent
