"""Game-agnostic AlphaZero core: network, MCTS, self-play, training, arena.

Nothing in this package imports a concrete game — only `game_interface.Game`.
A new game is supported by adding a plugin under `games/` and wiring it into
the orchestrator.
"""
