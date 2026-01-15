"""Wrapper to clip actions to valid range."""

class ActionClipWrapper:
    """Clip actions to the action space bounds."""
    
    def __init__(self, env):
        self.env = env
