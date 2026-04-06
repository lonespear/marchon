"""
play.py — Play against a trained Archon checkpoint in a browser.

Usage:
    python play.py --checkpoint checkpoints/archon_iter_0100.pt [--port 5001] [--sims 20]

Human can choose to play White, Black, or random at the start of each game.
Runs on port 5001 (separate from the training dashboard on port 5000).
"""
import argparse
import random
import sys
import threading

import chess
import torch
from flask import Flask, jsonify, render_template, request

from config import MarchonConfig, config
from env.chess_env import ChessEnv
from mcts.mcts import MCTS
from model.network import MarchonNet

# ── Module-level game state (single player, no sessions) ─────────────────────
board = chess.Board()
move_history: list[str] = []   # SAN strings
game_over = False
result_str = ""
network: MarchonNet | None = None
mcts_agent: MCTS | None = None
ckpt_meta: dict = {"iteration": 0, "elo": 0, "total_games": 0}
_move_lock = threading.Lock()
_cached_value: float = 0.0   # last eval; updated after each move, not on every poll
human_color: int = chess.WHITE  # chess.WHITE or chess.BLACK


# ── Checkpoint loading ───────────────────────────────────────────────────────
def load_checkpoint(path: str, sims_override: int | None = None) -> None:
    """Load ArchonNet weights and build MCTS agent."""
    global network, mcts_agent, ckpt_meta

    cfg = config  # use the module-level config instance

    try:
        ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    except FileNotFoundError:
        print(f"[play] ERROR: checkpoint not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Build network from config (must match checkpoint architecture)
    net = MarchonNet(
        num_planes     = cfg.num_planes,
        num_res_blocks = cfg.num_res_blocks,
        channels       = cfg.channels,
        num_moe_blocks = cfg.num_moe_blocks,
        num_experts    = cfg.num_experts,
        top_k          = cfg.top_k,
    )

    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(
            f"[play] ERROR: architecture mismatch loading checkpoint.\n"
            f"  {exc}\n"
            f"  Check that config.py matches the checkpoint's architecture.",
            file=sys.stderr,
        )
        sys.exit(1)

    net.to(cfg.device)
    net.eval()
    network = net

    # Optionally override simulation count for faster play
    if sims_override is not None:
        cfg.num_simulations = sims_override

    mcts_agent = MCTS(network, cfg)

    ckpt_meta = {
        "iteration": ckpt.get("iteration", 0),
        "elo": round(ckpt.get("elo", 0)),
        "total_games": ckpt.get("total_games", 0),
    }
    print(
        f"[play] Loaded checkpoint: iter={ckpt_meta['iteration']}, "
        f"elo={ckpt_meta['elo']}, sims={cfg.num_simulations}"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_value_estimate() -> float:
    """Return White-relative value in [-1, 1]. Returns 0.0 if game is over."""
    if game_over or network is None:
        return 0.0
    env = ChessEnv()
    env.board = board.copy()
    state = env.encode_state()
    mask = network.get_legal_mask(env)
    _, value = network.predict(state, mask, device=config.device)
    # network value is from current player's perspective → flip if Black to move
    if board.turn == chess.BLACK:
        value = -value
    return float(value)


def _detect_game_over() -> tuple[bool, str]:
    """Return (is_over, result_string)."""
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return True, "Game over"
        if outcome.winner is chess.WHITE:
            return True, "White wins"
        if outcome.winner is chess.BLACK:
            return True, "Black wins"
        return True, "Draw"
    return False, ""


def _get_status_message() -> str:
    """Human-readable status for display."""
    if game_over:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return "Game over."
        if outcome.winner is chess.WHITE:
            termination = outcome.termination
            if termination == chess.Termination.CHECKMATE:
                return "Checkmate! White wins."
            return f"White wins. ({termination.name.replace('_', ' ').title()})"
        if outcome.winner is chess.BLACK:
            termination = outcome.termination
            if termination == chess.Termination.CHECKMATE:
                return "Checkmate! Black wins."
            return f"Black wins. ({termination.name.replace('_', ' ').title()})"
        termination = outcome.termination
        return f"Draw. ({termination.name.replace('_', ' ').title()})"
    if board.turn == human_color:
        return "Your move"
    return "AI is thinking..."


# ── Flask app ─────────────────────────────────────────────────────────────────
def create_play_app() -> Flask:
    app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")

    @app.route("/")
    def index():
        return render_template(
            "play.html",
            iteration=ckpt_meta["iteration"],
            elo=ckpt_meta["elo"],
        )

    @app.route("/api/state")
    def api_state():
        return jsonify(
            {
                "fen": board.fen(),
                "moves": move_history,
                "status": _get_status_message(),
                "value": _cached_value,   # no NN call on poll — instant response
                "game_over": game_over,
                "result": result_str,
                "iteration": ckpt_meta["iteration"],
                "elo": ckpt_meta["elo"],
                "human_color": "white" if human_color == chess.WHITE else "black",
            }
        )

    @app.route("/api/move", methods=["POST"])
    def api_move():
        global board, move_history, game_over, result_str

        # Prevent concurrent MCTS calls
        acquired = _move_lock.acquire(blocking=False)
        if not acquired:
            return jsonify({"error": "AI is already thinking"}), 429

        try:
            global _cached_value
            if game_over:
                return jsonify({"error": "Game is already over"}), 400

            if board.turn != human_color:
                return jsonify({"error": "It is not your turn"}), 400

            data = request.get_json(force=True)
            uci_str = (data or {}).get("move", "")
            if not uci_str:
                return jsonify({"error": "No move provided"}), 400

            # Parse and validate human move
            try:
                human_move = chess.Move.from_uci(uci_str)
            except ValueError:
                return jsonify({"error": f"Invalid UCI: {uci_str}"}), 400

            if human_move not in board.legal_moves:
                return jsonify({"error": f"Illegal move: {uci_str}"}), 400

            # Get SAN *before* pushing (SAN depends on pre-push position)
            human_san = board.san(human_move)
            board.push(human_move)
            move_history.append(human_san)

            # Check if game ended after human move
            game_over, result_str = _detect_game_over()
            ai_move_uci = None
            ai_move_san = None

            if not game_over:
                # AI responds
                ai_move = mcts_agent.select_move(
                    board, temperature=0.0, add_noise=False
                )
                if ai_move is not None:
                    ai_move_san = board.san(ai_move)
                    ai_move_uci = ai_move.uci()
                    board.push(ai_move)
                    move_history.append(ai_move_san)
                    game_over, result_str = _detect_game_over()

            _cached_value = _get_value_estimate()

            return jsonify(
                {
                    "fen": board.fen(),
                    "ai_move": ai_move_uci,
                    "ai_move_san": ai_move_san,
                    "status": _get_status_message(),
                    "value": _cached_value,
                    "game_over": game_over,
                    "result": result_str,
                    "moves": move_history,
                }
            )

        finally:
            _move_lock.release()

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        global board, move_history, game_over, result_str, _cached_value, human_color
        data = request.get_json(force=True) or {}
        color_req = data.get("color", "white")
        if color_req == "random":
            color_req = random.choice(["white", "black"])
        human_color = chess.WHITE if color_req == "white" else chess.BLACK

        board = chess.Board()
        move_history = []
        game_over = False
        result_str = ""
        _cached_value = 0.0

        ai_move_uci = None
        ai_move_san = None

        # If human plays Black, AI (White) makes the opening move immediately
        if human_color == chess.BLACK:
            ai_move = mcts_agent.select_move(board, temperature=0.0, add_noise=False)
            if ai_move is not None:
                ai_move_san = board.san(ai_move)
                ai_move_uci = ai_move.uci()
                board.push(ai_move)
                move_history.append(ai_move_san)
                game_over, result_str = _detect_game_over()
                _cached_value = _get_value_estimate()

        return jsonify({
            "fen": board.fen(),
            "human_color": color_req,
            "ai_move": ai_move_uci,
            "ai_move_san": ai_move_san,
            "moves": move_history,
            "status": _get_status_message(),
            "value": _cached_value,
            "game_over": game_over,
        })

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Play chess against a trained Archon checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt checkpoint file"
    )
    parser.add_argument(
        "--port", type=int, default=5001, help="Port to serve on (default: 5001)"
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="Override number of MCTS simulations (default: from config)",
    )
    args = parser.parse_args()

    load_checkpoint(args.checkpoint, sims_override=args.sims)

    app = create_play_app()
    print(f"[play] Starting server at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
