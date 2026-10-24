"""pocket-recall: lossless, local long-term memory for a personal AI assistant."""

__all__ = ["MemoryStore", "Recall"]
__version__ = "0.2.0"


def __getattr__(name: str):
    # Lazy, so the hooks (which import only the standard library) start fast.
    if name in __all__:
        from . import store

        return getattr(store, name)
    raise AttributeError(name)
