"""Explanation pipeline."""

# Lazy imports to avoid requiring kedro for standalone testing
def create_pipeline(**kwargs):
    """Create explanation pipeline."""
    from .pipeline import create_pipeline as _create_pipeline
    return _create_pipeline(**kwargs)

__all__ = ["create_pipeline"]
