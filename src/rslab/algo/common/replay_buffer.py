"""Replay buffer for experience replay."""

class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, max_size):
        self.max_size = max_size
