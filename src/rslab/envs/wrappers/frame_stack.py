"""Wrapper to stack multiple frames."""

class FrameStackWrapper:
    """Stack multiple observation frames."""
    
    def __init__(self, env, num_frames=4):
        self.env = env
        self.num_frames = num_frames
