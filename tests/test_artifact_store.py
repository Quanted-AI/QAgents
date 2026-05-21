"""Tests for ArtifactStore and _NamespacedStore.

Validates dict-like access, typed get, version history, reserved key
prefix protection, SDK-internal bypass, and namespaced store delegation.
"""

import unittest

from quanted_agents.artifact_store import ArtifactStore, _NamespacedStore


class TestArtifactStore(unittest.TestCase):
    """Tests for ArtifactStore core operations."""

    def setUp(self) -> None:
        self.store = ArtifactStore()

    def test_set_and_get_item(self) -> None:
        """Store a value and retrieve it via bracket access."""
        self.store["key"] = "value"
        self.assertEqual(self.store["key"], "value")

    def test_set_overwrites_latest(self) -> None:
        """Writing the same key twice returns the latest value."""
        self.store["key"] = "first"
        self.store["key"] = "second"
        self.assertEqual(self.store["key"], "second")

    def test_get_missing_key_raises(self) -> None:
        """Accessing a nonexistent key raises KeyError."""
        with self.assertRaises(KeyError):
            _ = self.store["nonexistent"]

    def test_get_typed(self) -> None:
        """Typed get succeeds when value matches the expected type."""
        self.store["count"] = 42
        result = self.store.get("count", int)
        self.assertEqual(result, 42)

    def test_get_typed_wrong_type_raises(self) -> None:
        """Typed get raises TypeError when value does not match."""
        self.store["count"] = "not an int"
        with self.assertRaises(TypeError) as ctx:
            self.store.get("count", int)
        self.assertIn("Expected int", str(ctx.exception))
        self.assertIn("got str", str(ctx.exception))

    def test_history_returns_all_versions(self) -> None:
        """Writing the same key multiple times builds version history."""
        self.store["draft"] = "v1"
        self.store["draft"] = "v2"
        self.store["draft"] = "v3"
        self.assertEqual(self.store.history("draft"), ["v1", "v2", "v3"])

    def test_history_returns_copy(self) -> None:
        """Mutating the returned history list does not affect the store."""
        self.store["key"] = "value"
        history = self.store.history("key")
        history.append("injected")
        self.assertEqual(self.store.history("key"), ["value"])

    def test_history_missing_key_returns_empty(self) -> None:
        """History for a never-written key returns an empty list."""
        self.assertEqual(self.store.history("never_written"), [])

    def test_contains_and_keys_and_len(self) -> None:
        """Standard dict-like contains, keys, and len operations."""
        self.store["a"] = 1
        self.store["b"] = 2
        self.assertIn("a", self.store)
        self.assertNotIn("c", self.store)
        self.assertEqual(set(self.store.keys()), {"a", "b"})
        self.assertEqual(len(self.store), 2)

    def test_underscore_prefix_rejected(self) -> None:
        """User code cannot write keys starting with '_'."""
        with self.assertRaises(KeyError) as ctx:
            self.store["_internal"] = "forbidden"
        self.assertIn("reserved for SDK", str(ctx.exception))

    def test_sdk_set_bypasses_prefix_check(self) -> None:
        """SDK-internal _sdk_set writes '_' prefixed keys without error."""
        self.store._sdk_set("_meta", "allowed")
        self.assertEqual(self.store["_meta"], "allowed")
        self.assertEqual(self.store.history("_meta"), ["allowed"])


class TestNamespacedStore(unittest.TestCase):
    """Tests for _NamespacedStore delegation with key prefixing."""

    def setUp(self) -> None:
        self.parent = ArtifactStore()
        self.ns = _NamespacedStore(self.parent, "branch_0")

    def test_namespaced_store_prefixes_keys(self) -> None:
        """Writes through _NamespacedStore appear in parent under prefix/key."""
        self.ns["result"] = "data"
        self.assertEqual(self.parent["branch_0/result"], "data")

    def test_namespaced_store_read(self) -> None:
        """Reading through namespaced store retrieves the prefixed key."""
        self.ns["output"] = 42
        self.assertEqual(self.ns["output"], 42)

    def test_namespaced_store_get_typed(self) -> None:
        """Typed access through namespaced view works correctly."""
        self.ns["value"] = 100
        result = self.ns.get("value", int)
        self.assertEqual(result, 100)

    def test_namespaced_store_get_typed_wrong_type(self) -> None:
        """Typed access raises TypeError for wrong type through namespace."""
        self.ns["value"] = "text"
        with self.assertRaises(TypeError):
            self.ns.get("value", int)

    def test_namespaced_store_history(self) -> None:
        """History through namespaced view returns correct versions."""
        self.ns["draft"] = "v1"
        self.ns["draft"] = "v2"
        self.assertEqual(self.ns.history("draft"), ["v1", "v2"])
        self.assertEqual(self.parent.history("branch_0/draft"), ["v1", "v2"])

    def test_namespaced_store_contains(self) -> None:
        """Containment check through namespaced view uses prefixed key."""
        self.ns["exists"] = True
        self.assertIn("exists", self.ns)
        self.assertNotIn("missing", self.ns)
        self.assertIn("branch_0/exists", self.parent)

    def test_multiple_namespaces_isolated(self) -> None:
        """Different namespaces in the same parent do not collide."""
        ns1 = _NamespacedStore(self.parent, "branch_0")
        ns2 = _NamespacedStore(self.parent, "branch_1")
        ns1["key"] = "from_0"
        ns2["key"] = "from_1"
        self.assertEqual(ns1["key"], "from_0")
        self.assertEqual(ns2["key"], "from_1")
        self.assertEqual(self.parent["branch_0/key"], "from_0")
        self.assertEqual(self.parent["branch_1/key"], "from_1")


if __name__ == "__main__":
    unittest.main()
