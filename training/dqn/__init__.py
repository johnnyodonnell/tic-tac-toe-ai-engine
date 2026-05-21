"""Model-free DQN bot: a Q-network trained by self-play, no search.

Unlike the AlphaZero core, this plays with a single forward pass — board to
Q-values to argmax. It shares the game interface and game plugins, nothing else.
"""
