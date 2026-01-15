"""Timing utilities for profiling."""

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name):
        self.name = name
