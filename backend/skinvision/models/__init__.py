__all__ = ["LMDAUNet"]


def __getattr__(name):
    if name == "LMDAUNet":
        from .networks import LMDAUNet
        return LMDAUNet
    raise AttributeError(name)
