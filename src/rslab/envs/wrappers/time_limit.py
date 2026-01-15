"""Wrapper to add time limit to episodes."""

class TimeLimitWrapper:
    """Add time limit to episodes."""
    
    def __init__(self, env, max_steps=500):
        self.env = env
        self.max_steps = max_steps
