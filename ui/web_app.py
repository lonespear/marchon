"""
web_app.py — Flask web dashboard for Archon.

Serves a single-page app at / that streams live training state via SSE.
Also exposes /api/game/<id> to fetch a historical game board.
"""

from flask import Flask, Response, jsonify, render_template
import chess
import chess.svg
import json
import time

from ui.dashboard import SharedState


BOARD_COLORS = {
    'square light':           '#f0d9b5',
    'square dark':            '#b58863',
    'square light lastmove':  '#cdd26a',
    'square dark lastmove':   '#aaa23a',
    'coord':                  '#e8c792',
    'margin':                 '#0f0f1e',
}


def create_app(shared_state: SharedState) -> Flask:
    app = Flask(__name__, template_folder='templates')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/stream')
    def stream():
        """SSE endpoint — pushes state JSON every 1 second."""
        def generate():
            try:
                while True:
                    s = shared_state.snapshot()
                    svg = chess.svg.board(
                        s['board'],
                        lastmove=s.get('last_move'),
                        size=400,
                        colors=BOARD_COLORS,
                    )
                    payload = {**s, 'board_svg': svg}
                    payload.pop('board')        # chess.Board not JSON-serialisable
                    payload['last_move'] = None  # chess.Move not JSON-serialisable
                    yield f"data: {json.dumps(payload)}\n\n"
                    time.sleep(1)
            except GeneratorExit:
                pass

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

    @app.route('/api/game/<int:game_id>')
    def get_game(game_id):
        """Return the final board SVG + moves for a historical game."""
        s = shared_state.snapshot()
        game = next((g for g in s['games'] if g['id'] == game_id), None)
        if not game:
            return jsonify({'error': 'not found'}), 404

        board = chess.Board()
        last_move = None
        for san in game['moves']:
            try:
                last_move = board.push_san(san)
            except Exception:
                break

        svg = chess.svg.board(board, lastmove=last_move, size=400, colors=BOARD_COLORS)
        return jsonify({
            'board_svg': svg,
            'moves':     game['moves'],
            'result':    game['result'],
        })

    @app.route('/api/game/<int:game_id>/ply/<int:ply>')
    def get_game_ply(game_id, ply):
        """Return board SVG at a specific ply (half-move) within a game."""
        s = shared_state.snapshot()
        game = next((g for g in s['games'] if g['id'] == game_id), None)
        if not game:
            return jsonify({'error': 'not found'}), 404

        board = chess.Board()
        last_move = None
        moves_played = game['moves'][:ply]
        for san in moves_played:
            try:
                last_move = board.push_san(san)
            except Exception:
                break

        svg = chess.svg.board(board, lastmove=last_move, size=400, colors=BOARD_COLORS)
        return jsonify({
            'board_svg':   svg,
            'moves':       moves_played,
            'ply':         ply,
            'total_plies': len(game['moves']),
        })

    return app
