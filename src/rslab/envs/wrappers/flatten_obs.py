"""Wrapper to flatten observations."""

class FlattenObsWrapper:
    """Flatten observation from dict or tuple to 1D array."""
    
    def __init__(self, env):
        self.env = env
