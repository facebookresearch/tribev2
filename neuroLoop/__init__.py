"""neuroLoop — SDK for working with TRIBE v2 brain predictions."""


def __getattr__(name):
    if name == "BrainAtlas":
        from neuroLoop.atlas import BrainAtlas
        return BrainAtlas
    raise AttributeError(f"module 'neuroLoop' has no attribute {name!r}")


__all__ = ["BrainAtlas"]
