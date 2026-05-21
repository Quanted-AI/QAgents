"""ArtifactStore: Typed key-value store with version history for workflow artifacts."""

from collections import defaultdict
from collections.abc import KeysView
from typing import Any, TypeVar

T = TypeVar("T")


class ArtifactStore:
    """Typed key-value store with version history for workflow artifacts.

    Provides dict-like access for the latest value per key and append-only
    version history. Keys starting with '_' are reserved for SDK internal
    use and cannot be written by user code.

    Example:
        store = ArtifactStore()
        store["analysis"] = AnalysisResult(...)
        store["analysis"] = RefinedAnalysis(...)  # overwrites latest, appends to history

        latest = store["analysis"]                     # RefinedAnalysis
        typed = store.get("analysis", RefinedAnalysis)  # typed access
        all_versions = store.history("analysis")        # [AnalysisResult, RefinedAnalysis]
    """

    def __init__(self) -> None:
        """Initialize an empty ArtifactStore."""
        self._data: dict[str, Any] = {}
        self._history: dict[str, list[Any]] = defaultdict(list)

    def __setitem__(self, key: str, value: Any) -> None:
        """Store a value, appending to version history.

        Args:
            key: The artifact key. Must not start with '_' (reserved for SDK).
            value: The artifact value to store.

        Raises:
            KeyError: If key starts with '_' (reserved for SDK internal use).
        """
        if key.startswith("_"):
            raise KeyError(
                f"Key '{key}' starts with '_' which is reserved for SDK internal use. "
                f"Use a key without the '_' prefix."
            )
        self._data[key] = value
        self._history[key].append(value)

    def __getitem__(self, key: str) -> Any:
        """Get the latest value for a key.

        Args:
            key: The artifact key to look up.

        Returns:
            The most recently stored value for this key.

        Raises:
            KeyError: If the key does not exist.
        """
        return self._data[key]

    def get(self, key: str, type_: type[T]) -> T:
        """Get the latest value for a key with runtime type checking.

        Args:
            key: The artifact key to look up.
            type_: The expected type. A runtime isinstance check is performed.

        Returns:
            The most recently stored value, typed as T.

        Raises:
            KeyError: If the key does not exist.
            TypeError: If the stored value is not an instance of type_.
        """
        value = self._data[key]
        if not isinstance(value, type_):
            raise TypeError(
                f"Expected {type_.__name__} for key '{key}', "
                f"got {type(value).__name__}"
            )
        return value

    def history(self, key: str) -> list[Any]:
        """Get the full version history for a key.

        Returns a COPY of the internal history list to prevent external mutation.

        Args:
            key: The artifact key.

        Returns:
            A list of all values stored under this key, in chronological order.
            Returns an empty list if the key has never been written.
        """
        return list(self._history[key])

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the store.

        Args:
            key: The artifact key.

        Returns:
            True if the key has been written at least once.
        """
        return key in self._data

    def keys(self) -> KeysView[str]:
        """Return all keys in the store.

        Returns:
            A view of all artifact keys.
        """
        return self._data.keys()

    def __len__(self) -> int:
        """Return the number of keys in the store.

        Returns:
            The count of distinct keys.
        """
        return len(self._data)

    def __bool__(self) -> bool:
        """An ArtifactStore is always truthy, even when empty.

        This prevents falsy-empty-store gotchas in ``store or default``
        expressions used by orchestration patterns.

        Returns:
            Always True.
        """
        return True

    def _sdk_set(self, key: str, value: Any) -> None:
        """SDK-internal method to write to reserved '_' prefix keys.

        Bypasses the '_' prefix restriction. Only called by orchestration
        patterns when trace_artifacts=True.

        Args:
            key: The artifact key (should start with '_').
            value: The value to store.
        """
        self._data[key] = value
        self._history[key].append(value)


class _NamespacedStore:
    """Namespaced view into an ArtifactStore with automatic key prefixing.

    Used by Parallel to give each branch an isolated key namespace within
    the shared parent store. All reads and writes are transparently prefixed
    with the branch identifier.

    Example:
        parent = ArtifactStore()
        branch_0 = _NamespacedStore(parent, "branch_0")
        branch_0["results"] = data  # actually writes to parent["branch_0/results"]
    """

    def __init__(self, parent: ArtifactStore, prefix: str) -> None:
        """Create a namespaced view of a parent ArtifactStore.

        Args:
            parent: The underlying ArtifactStore to delegate to.
            prefix: The namespace prefix (e.g., "branch_0").
        """
        self._parent: ArtifactStore = parent
        self._prefix: str = prefix

    def _prefixed(self, key: str) -> str:
        """Create the prefixed key.

        Args:
            key: The unprefixed key.

        Returns:
            The key with namespace prefix applied.
        """
        return f"{self._prefix}/{key}"

    def __setitem__(self, key: str, value: Any) -> None:
        """Store a value under the namespaced key.

        Args:
            key: The unprefixed key.
            value: The value to store.

        Raises:
            KeyError: If the prefixed key starts with '_'.
        """
        self._parent[self._prefixed(key)] = value

    def __getitem__(self, key: str) -> Any:
        """Get the latest value for a namespaced key.

        Args:
            key: The unprefixed key.

        Returns:
            The most recently stored value.

        Raises:
            KeyError: If the prefixed key does not exist.
        """
        return self._parent[self._prefixed(key)]

    def get(self, key: str, type_: type[T]) -> T:
        """Get the latest value with runtime type checking.

        Args:
            key: The unprefixed key.
            type_: The expected type.

        Returns:
            The typed value.

        Raises:
            KeyError: If the prefixed key does not exist.
            TypeError: If the value is not an instance of type_.
        """
        return self._parent.get(self._prefixed(key), type_)

    def history(self, key: str) -> list[Any]:
        """Get the version history for a namespaced key.

        Args:
            key: The unprefixed key.

        Returns:
            A list of all values in chronological order.
        """
        return self._parent.history(self._prefixed(key))

    def __contains__(self, key: str) -> bool:
        """Check if a namespaced key exists.

        Args:
            key: The unprefixed key.

        Returns:
            True if the prefixed key exists in the parent store.
        """
        return self._prefixed(key) in self._parent
