from ragbot.cache import LRUCache, normalize


def test_normalize_whitespace_and_case():
    assert normalize("  What   IS RAG? ") == "what is rag?"


def test_hit_and_miss():
    c = LRUCache(2)
    c.put("a b", 1)
    assert c.get("A   B") == 1
    assert c.get("missing") is None


def test_evicts_least_recently_used():
    c = LRUCache(2)
    c.put("one", 1)
    c.put("two", 2)
    c.get("one")  # "two" is now the oldest
    c.put("three", 3)
    assert c.get("two") is None
    assert c.get("one") == 1 and c.get("three") == 3
    assert len(c) == 2


def test_zero_size_disables_cache():
    c = LRUCache(0)
    c.put("x", 1)
    assert c.get("x") is None
