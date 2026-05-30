__all__ = ["Voyager"]


def __getattr__(name):
    if name == "Voyager":
        from .voyager import Voyager

        return Voyager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
