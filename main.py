"""
main.py — Marchon entry point.

Usage
-----
  # Start training with the live web dashboard (default)
  python main.py

  # Train silently, logging to marchon.log
  python main.py --headless

  # Resume from a checkpoint
  python main.py --checkpoint checkpoints/marchon_iter_0050.pt

  # Limit iterations
  python main.py --iterations 500

Architecture note
-----------------
In interactive mode the process has two threads:

  Main thread  -> runs Flask (blocks on flask_app.run)
  Train thread -> runs Trainer.run()  (daemon, killed on Ctrl+C)

They share a SharedState object.  The trainer writes to it; Flask
serves it to browsers via SSE every second.
"""

import argparse
import logging
import threading

from config import config
from training.trainer import Trainer
from ui.dashboard import SharedState
from ui.web_app import create_app


class SharedStateHandler(logging.Handler):
    """Forwards every log record into SharedState so the TUI can display it."""

    def __init__(self, shared_state):
        super().__init__()
        self._shared = shared_state
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._shared.push_log(self.format(record))
        except Exception:
            pass


def setup_logging(shared_state=None):
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
        handlers= [
            logging.FileHandler(config.log_file, encoding="utf-8"),
        ],
    )
    if shared_state is not None:
        logging.getLogger().addHandler(SharedStateHandler(shared_state))
    else:
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        logging.getLogger().addHandler(console)


def main():
    parser = argparse.ArgumentParser(
        description="Marchon — MoE Chess RL Training"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Train without the TUI (useful over SSH when you just want logs)",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of training iterations (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a checkpoint .pt file to resume from",
    )
    args = parser.parse_args()

    shared  = SharedState()
    setup_logging(shared_state=None if args.headless else shared)
    trainer = Trainer(config, shared_state=shared)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if args.headless:
        trainer.run(num_iterations=args.iterations)

    else:
        stop_event = threading.Event()

        train_thread = threading.Thread(
            target = trainer.run,
            kwargs = {"num_iterations": args.iterations, "stop_event": stop_event},
            daemon = True,
            name   = "MarchonTrainer",
        )
        train_thread.start()

        flask_app = create_app(shared)
        print(f"\n  Dashboard -> http://localhost:5000")
        print(f"  On network -> http://<server-ip>:5000\n")
        try:
            flask_app.run(host='0.0.0.0', port=5000, debug=False,
                          use_reloader=False, threaded=True)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            train_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
