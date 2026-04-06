"""
dashboard.py — Archon TUI monitor built with Textual.

Layout
──────
┌─────────────────────────┬──────────────────────────┐
│                         │                          │
│   Chess Board           │   Move History           │
│   (upper-left)          │   (upper-right)          │
│                         │                          │
├─────────────────────────┼──────────────────────────┤
│                         │                          │
│   Training Loss Graph   │   Win/Loss/Draw + ELO    │
│   (lower-left)          │   (lower-right)          │
│                         │                          │
└──────────────────────── ┴──────────────────────────┘
│  ◄  Game 1  Game 2  Game 3  …  (horizontal scroll) │
└─────────────────────────────────────────────────────┘

The dashboard polls a SharedState object every second.
SharedState is written by the trainer thread (thread-safe via a Lock)
and read by the UI thread via snapshot().
Press 'r' to snap back to the live game after clicking a historical one.
Press 'q' to quit.
"""

import threading
from typing import List, Optional

import chess
from rich.markup import escape as _markup_escape
from rich.text import Text


from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, Static


# ── Shared state (trainer → UI bridge) ────────────────────────────────────────

class SharedState:
    """
    Thread-safe container for all data the UI needs to display.

    The trainer calls push_game / update_* from its own thread.
    The UI calls snapshot() every second to get a consistent copy.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.current_board: chess.Board = chess.Board()
        self.current_moves: List[str]   = []
        self.current_result: Optional[str] = None
        self.current_last_move: Optional[chess.Move] = None
        self.iteration: int = 0
        self.buffer_size: int = 0

        self.policy_losses:   List[float] = []
        self.value_losses:    List[float] = []
        self.combined_losses: List[float] = []

        self.wins   = 0
        self.draws  = 0
        self.losses = 0

        self.elo_history: List[float] = [1000.0]
        self.games: List[dict] = []
        self.log_messages: List[str] = []

    # ── Trainer-side writes ───────────────────────────────────────────────────

    def push_log(self, msg: str) -> None:
        with self._lock:
            self.log_messages.append(msg)
            if len(self.log_messages) > 200:
                self.log_messages = self.log_messages[-200:]

    def push_game(self, record) -> None:
        with self._lock:
            # Reconstruct final board, capturing last move
            board = chess.Board()
            last_move = None
            for san in record.moves:
                try:
                    last_move = board.push_san(san)
                except Exception:
                    break

            self.current_board     = board
            self.current_moves     = record.moves.copy()
            self.current_result    = record.result
            self.current_last_move = last_move

            self.games.append({
                "id":       record.game_id,
                "result":   record.result,
                "moves":    record.moves.copy(),
                "num_moves": record.num_moves,
            })

    def update_stats(self, wins: int, draws: int, losses: int) -> None:
        with self._lock:
            self.wins   += wins
            self.draws  += draws
            self.losses += losses

    def update_losses(self, policy, value, combined) -> None:
        with self._lock:
            self.policy_losses   = list(policy)
            self.value_losses    = list(value)
            self.combined_losses = list(combined)

    def update_elo(self, new_elo: float) -> None:
        with self._lock:
            self.elo_history.append(new_elo)

    def update_live_board(self, board, moves, last_move=None) -> None:
        """Called by SelfPlay after every move — enables live board updates."""
        with self._lock:
            self.current_board     = board.copy()
            self.current_moves     = list(moves)
            self.current_result    = None
            self.current_last_move = last_move

    def update_iteration(self, iteration: int, buffer_size: int) -> None:
        """Called by Trainer at the start of each iteration for the status bar."""
        with self._lock:
            self.iteration   = iteration
            self.buffer_size = buffer_size

    # ── UI-side reads ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a deep-enough copy for safe single-threaded reading."""
        with self._lock:
            return {
                "board":           self.current_board.copy(),
                "moves":           list(self.current_moves),
                "result":          self.current_result,
                "last_move":       self.current_last_move,
                "iteration":       self.iteration,
                "buffer_size":     self.buffer_size,
                "policy_losses":   list(self.policy_losses),
                "value_losses":    list(self.value_losses),
                "combined_losses": list(self.combined_losses),
                "wins":            self.wins,
                "draws":           self.draws,
                "losses":          self.losses,
                "elo_history":     list(self.elo_history),
                "games":           [dict(g) for g in self.games],
                "log_messages":    list(self.log_messages),
            }


# ── Chess piece unicode map & board colours ────────────────────────────────────

_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

# Classic wood-tone chess board palette
_SQ_LIGHT    = "on #f0d9b5"   # cream
_SQ_DARK     = "on #b58863"   # warm brown
_SQ_HI_LIGHT = "on #cdd26a"   # last-move highlight (light square)
_SQ_HI_DARK  = "on #aaa23a"   # last-move highlight (dark square)
_WP          = "bold bright_white"
_BP          = "bold #140a00"


# ── Widgets ───────────────────────────────────────────────────────────────────

class ChessBoardWidget(Static):
    """
    Large chess board rendered with 2-row ranks and 5-wide squares.

    Each square is 5 terminal columns wide; each rank is 2 rows tall,
    giving roughly 3× the area of a compact 3-wide / 1-row board.
    Classic wood-tone palette matches the overall dark UI theme.
    """

    DEFAULT_CSS = """
    ChessBoardWidget {
        border: solid #7b8cde;
        padding: 1 0;
        height: 100%;
    }
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._board: chess.Board               = chess.Board()
        self._last_move: Optional[chess.Move]  = None

    def set_position(self, board: chess.Board, last_move: Optional[chess.Move] = None):
        self._board     = board
        self._last_move = last_move
        self.update(self._build_board())

    def on_mount(self):
        self.update(self._build_board())

    def _build_board(self) -> Text:
        t  = Text()
        hi = {self._last_move.from_square, self._last_move.to_square} \
             if self._last_move else set()

        # 7-wide squares: each file label centred in 7 chars
        FILES = "".join(f"   {f}   " for f in "abcdefgh")
        t.append(f"     {FILES}\n", style="bold #e8c792")

        for rank in range(7, -1, -1):
            # ── top padding row (background colour only) ─────────────────────
            t.append("     ", style="dim")
            for file in range(8):
                sq    = chess.square(file, rank)
                light = (rank + file) % 2 == 1
                bg    = (_SQ_HI_LIGHT if sq in hi else _SQ_LIGHT) if light \
                        else (_SQ_HI_DARK if sq in hi else _SQ_DARK)
                t.append("       ", style=bg)   # 7 spaces
            t.append("\n")

            # ── piece row ───────────────────────────────────────────────────
            t.append(f" {rank + 1}   ", style="bold #e8c792")   # 5-char label
            for file in range(8):
                sq    = chess.square(file, rank)
                piece = self._board.piece_at(sq)
                light = (rank + file) % 2 == 1
                bg    = (_SQ_HI_LIGHT if sq in hi else _SQ_LIGHT) if light \
                        else (_SQ_HI_DARK if sq in hi else _SQ_DARK)
                if piece:
                    sym = _PIECES.get(piece.symbol(), "?")
                    fg  = _WP if piece.color == chess.WHITE else _BP
                    t.append(f"   {sym}   ", style=f"{fg} {bg}")  # 7-wide
                else:
                    t.append("       ", style=bg)   # 7 spaces
            t.append(f" {rank + 1}\n", style="bold #e8c792")

        t.append(f"     {FILES}\n", style="bold #e8c792")
        return t


class MoveHistoryWidget(ScrollableContainer):
    """
    Two-column move list — properly scrollable via keyboard and mouse.

    Uses a ScrollableContainer so the user can scroll up/down through
    the full game history when it grows beyond the panel height.
    """

    DEFAULT_CSS = """
    MoveHistoryWidget {
        border: solid #7b8cde;
        height: 100%;
    }
    MoveHistoryWidget Static {
        width: 100%;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("No moves yet.", id="moves-inner")

    def set_moves(self, moves: List[str], result: Optional[str] = None):
        inner = self.query_one("#moves-inner", Static)
        if not moves:
            inner.update("No moves yet.")
            return

        t = Text()
        t.append("Move History\n", style="bold #7b8cde")
        t.append("─" * 24 + "\n", style="dim")

        for i in range(0, len(moves), 2):
            num   = i // 2 + 1
            white = moves[i]
            black = moves[i + 1] if i + 1 < len(moves) else ""
            t.append(f"{num:>3}. ", style="dim")
            t.append(f"{white:<10}", style="bold bright_white")
            t.append(f"{black}\n",  style="bold #c8c8c8")

        if result:
            label = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}.get(result, "*")
            color = {"white": "bright_green", "black": "bright_red", "draw": "#e8c792"}.get(result, "white")
            t.append(f"\nResult: {label}", style=f"bold {color}")

        inner.update(t)
        self.scroll_end(animate=False)


class TrainingGraphWidget(Static):
    """
    ASCII line chart for policy loss, value loss, and combined loss.

    Each series is down-sampled to fit the available terminal width,
    then plotted row-by-row with unicode block characters.
    """

    DEFAULT_CSS = """
    TrainingGraphWidget {
        border: solid #c8963e;
        padding: 1 2;
        height: 100%;
    }
    """

    _SERIES_COLORS = {
        "Policy": "bright_yellow",
        "Value":  "bright_cyan",
        "Total":  "bright_magenta",
    }

    def __init__(self, **kwargs):
        super().__init__("Waiting for training to start…", **kwargs)

    def set_losses(self, policy, value, combined):
        t = Text()
        t.append("  Training Loss\n", style="bold underline #ffff00")
        t.append("  " + "─" * 22 + "\n", style="dim")

        series = {"Policy": policy, "Value": value, "Total": combined}
        chart  = _ascii_chart(series, width=46, height=11)
        t.append_text(chart)

        if policy:
            t.append(
                f"\n  Latest - Policy: {policy[-1]:.4f}  "
                f"Value: {value[-1]:.4f}  "
                f"Total: {combined[-1]:.4f}",
                style="dim",
            )
        self.update(t)


class StatsWidget(Static):
    """Win/draw/loss bar, ELO readout, and ELO sparkline."""

    DEFAULT_CSS = """
    StatsWidget {
        border: solid #4caf50;
        padding: 1 2;
        height: 100%;
    }
    """

    def __init__(self, **kwargs):
        super().__init__("Waiting for game data…", **kwargs)

    def set_stats(self, wins: int, draws: int, losses: int, elo_history: List[float]):
        total = wins + draws + losses
        t = Text()
        t.append("  Performance\n", style="bold underline #00ffff")
        t.append("  " + "─" * 22 + "\n", style="dim")

        # Win/draw/loss bar ────────────────────────────────────────────────────
        if total > 0:
            BAR = 32
            w = max(1, round(wins   / total * BAR))
            d = max(0, round(draws  / total * BAR))
            l = max(0, BAR - w - d)
            t.append("  ")
            t.append("█" * w,           style="bright_green")
            t.append("█" * d,           style="yellow")
            t.append("█" * l,           style="bright_red")
            win_pct = wins / total * 100
            t.append(
                f"  {wins}W  {draws}D  {losses}L  "
                f"({total} games,  {win_pct:.1f}% wins)\n",
                style="dim",
            )
        else:
            t.append("  No games played yet.\n", style="dim")

        # ELO ──────────────────────────────────────────────────────────────────
        elo     = elo_history[-1] if elo_history else 1000.0
        delta   = elo_history[-1] - elo_history[-2] if len(elo_history) >= 2 else 0.0
        d_color = "bright_green" if delta >= 0 else "bright_red"
        d_sign  = "+" if delta >= 0 else ""

        t.append(f"\n  ELO: ", style="bold")
        t.append(f"{elo:.0f}", style="bold bright_cyan")
        t.append(f"  ({d_sign}{delta:.0f})\n", style=d_color)

        # ELO sparkline ────────────────────────────────────────────────────────
        if len(elo_history) > 1:
            t.append("  ", style="dim")
            t.append_text(_sparkline(elo_history[-50:], "bright_cyan"))
            t.append("\n")

        self.update(t)


class GameScrollWidget(ScrollableContainer):
    """Horizontally scrollable row of game buttons at the bottom."""

    DEFAULT_CSS = """
    GameScrollWidget {
        height: 100%;
        border: solid #888888;
        layout: horizontal;
        overflow-x: scroll;
        overflow-y: hidden;
        padding: 0 1;
    }
    Button {
        min-width: 12;
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("  No games yet  ", id="no-games-label")

    def add_game(self, game_id: int, result: str, num_moves: int) -> None:
        sym   = {"white": "1-0", "black": "0-1", "draw": "½-½"}.get(result, "?")
        label = f"G{game_id} {sym}"
        color = {"white": "success", "black": "error", "draw": "warning"}.get(result, "default")
        btn   = Button(label, id=f"game-{game_id}", variant=color)  # type: ignore[arg-type]

        try:
            self.query_one("#no-games-label").remove()
        except Exception:
            pass

        self.mount(btn)
        self.scroll_end(animate=False)


class LogWidget(ScrollableContainer):
    """
    Live training log — streams messages from the Python logging system.

    The trainer pushes log lines into SharedState; the dashboard reads them
    every second and appends new entries here.  Scroll up to review history;
    the view auto-scrolls to the bottom whenever new lines arrive.
    """

    DEFAULT_CSS = """
    LogWidget {
        border: solid #444466;
        height: 100%;
        overflow-y: auto;
        padding: 0 1;
    }
    LogWidget Label {
        width: 100%;
    }
    """

    # Colour-code by log level keyword in the message
    _LEVEL_COLORS = {
        "ERROR":   "bright_red",
        "WARNING": "yellow",
        "INFO":    "dim",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shown = 0          # how many lines already mounted

    def append_lines(self, messages: List[str]) -> None:
        """Mount only the lines we haven't shown yet, then scroll down."""
        new = messages[self._shown:]
        if not new:
            return
        for msg in new:
            color = "dim"
            for level, c in self._LEVEL_COLORS.items():
                if level in msg:
                    color = c
                    break
            self.mount(Label(f"[{color}]{_markup_escape(msg)}[/]", classes="log-line"))
        self._shown += len(new)
        self.scroll_end(animate=False)


# ── ASCII chart helpers ────────────────────────────────────────────────────────

_CHART_COLORS = ["bright_yellow", "bright_cyan", "bright_magenta"]


def _ascii_chart(series: dict, width: int = 44, height: int = 10) -> Text:
    """Render multiple named series as an ASCII line chart."""
    t = Text()

    all_vals = [v for vals in series.values() for v in vals]
    if not all_vals:
        t.append("  No data yet.\n", style="dim")
        return t

    mn, mx = min(all_vals), max(all_vals)
    span   = max(mx - mn, 1e-9)

    # Build character grid (rows × cols)
    grid: List[List] = [[" "] * width for _ in range(height)]

    for (label, vals), color in zip(series.items(), _CHART_COLORS):
        if len(vals) < 2:
            continue
        step    = max(1, len(vals) // width)
        sampled = vals[::step][-width:]
        for x, v in enumerate(sampled):
            y = int((v - mn) / span * (height - 1))
            y = (height - 1) - y   # flip: high value → low row index
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = (color, "•")

    for row_i, row in enumerate(grid):
        if row_i % 3 == 0:
            val = mx - (row_i / max(height - 1, 1)) * span
            t.append(f"  {val:6.3f}│", style="dim")
        else:
            t.append("         │", style="dim")
        for cell in row:
            if isinstance(cell, tuple):
                t.append(cell[1], style=cell[0])
            else:
                t.append(cell)
        t.append("\n")

    t.append("         └" + "─" * width + "\n", style="dim")

    # Legend
    t.append("  ")
    for (label, _), color in zip(series.items(), _CHART_COLORS):
        t.append(f"• {label}  ", style=color)
    t.append("\n")
    return t


def _sparkline(values: List[float], color: str = "bright_cyan") -> Text:
    CHARS = "▁▂▃▄▅▆▇█"
    t = Text()
    if len(values) < 2:
        return t
    mn, mx = min(values), max(values)
    span   = max(mx - mn, 1e-9)
    for v in values:
        idx = int((v - mn) / span * (len(CHARS) - 1))
        t.append(CHARS[idx], style=color)
    return t


# ── Main app ──────────────────────────────────────────────────────────────────

class ArchonDashboard(App):
    """
    Marchon training monitor.

    Keybindings
    -----------
    q : quit
    r : return to live game view (after clicking a historical game)
    """

    TITLE = "Marchon — MoE Chess RL Monitor"

    CSS = """
    Screen {
        layout: vertical;
        background: #0f0f1e;
    }
    Header {
        background: #16213e;
        color: #e8c792;
        text-style: bold;
    }
    Footer {
        background: #16213e;
        color: #7b8cde;
    }
    #top-row {
        height: 56%;
        layout: horizontal;
    }
    #board-col {
        width: 70%;
    }
    #moves-col {
        width: 30%;
    }
    #bottom-row {
        height: 19%;
        layout: horizontal;
    }
    #graph-col {
        width: 50%;
    }
    #stats-col {
        width: 50%;
    }
    #game-scroll {
        height: 9%;
    }
    #log-row {
        height: 16%;
    }
    """

    BINDINGS = [
        ("q", "quit",       "Quit"),
        ("r", "live_view",  "Live view"),
        ("l", "toggle_log", "Log"),
    ]

    def __init__(self, shared_state: SharedState, **kwargs):
        super().__init__(**kwargs)
        self.shared          = shared_state
        self._known_games    = 0
        self._viewing_live   = True
        self._log_visible    = True

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            with Container(id="board-col"):
                yield ChessBoardWidget(id="board")
            with Container(id="moves-col"):
                yield MoveHistoryWidget(id="moves")
        with Horizontal(id="bottom-row"):
            with Container(id="graph-col"):
                yield TrainingGraphWidget(id="graph")
            with Container(id="stats-col"):
                yield StatsWidget(id="stats")
        yield GameScrollWidget(id="game-scroll")
        yield LogWidget(id="log-row")
        yield Footer()

    def on_mount(self):
        self.set_interval(1.0, self._refresh)

    # ── Periodic refresh ──────────────────────────────────────────────────────

    def _refresh(self):
        state = self.shared.snapshot()

        # Only update board + moves if we're in live view
        if self._viewing_live:
            self.query_one("#board", ChessBoardWidget).set_position(state["board"])
            self.query_one("#moves", MoveHistoryWidget).set_moves(
                state["moves"], state["result"]
            )

        self.query_one("#graph", TrainingGraphWidget).set_losses(
            state["policy_losses"],
            state["value_losses"],
            state["combined_losses"],
        )
        self.query_one("#stats", StatsWidget).set_stats(
            state["wins"], state["draws"], state["losses"],
            state["elo_history"],
        )

        # Add newly finished game buttons
        games = state["games"]
        if len(games) > self._known_games:
            scroll = self.query_one("#game-scroll", GameScrollWidget)
            for g in games[self._known_games:]:
                scroll.add_game(g["id"], g["result"], g["num_moves"])
            self._known_games = len(games)

        # Stream new log lines
        self.query_one("#log-row", LogWidget).append_lines(state["log_messages"])

    # ── Game history navigation ───────────────────────────────────────────────

    @on(Button.Pressed)
    def _on_game_button(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if not btn_id.startswith("game-"):
            return

        game_id = int(btn_id.split("-", 1)[1])
        state   = self.shared.snapshot()
        game    = next((g for g in state["games"] if g["id"] == game_id), None)
        if game is None:
            return

        # Replay moves to reconstruct board
        board = chess.Board()
        for san in game["moves"]:
            try:
                board.push_san(san)
            except Exception:
                break

        self._viewing_live = False
        self.query_one("#board", ChessBoardWidget).set_position(board)
        self.query_one("#moves", MoveHistoryWidget).set_moves(
            game["moves"], game["result"]
        )

    def action_live_view(self) -> None:
        """Snap back to the most recent live game."""
        self._viewing_live = True
        state = self.shared.snapshot()
        self.query_one("#board", ChessBoardWidget).set_position(state["board"])
        self.query_one("#moves", MoveHistoryWidget).set_moves(
            state["moves"], state["result"]
        )

    def action_toggle_log(self) -> None:
        """Show/hide the log panel (press 'l')."""
        log = self.query_one("#log-row", LogWidget)
        self._log_visible = not self._log_visible
        log.display = self._log_visible
