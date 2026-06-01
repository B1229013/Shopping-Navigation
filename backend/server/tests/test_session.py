from server.session import Session, SessionStore


def test_create_session_assigns_unique_id():
    store = SessionStore()
    s1 = store.create(goal="find milk", goal_objects=["milk"])
    s2 = store.create(goal="find bread", goal_objects=["bread"])
    assert s1.id != s2.id
    assert s1.goal == "find milk"
    assert s1.arrived is False
    assert s1.pending_question is None


def test_get_returns_same_session():
    store = SessionStore()
    s = store.create(goal="g", goal_objects=[])
    assert store.get(s.id) is s


def test_get_unknown_id_returns_none():
    store = SessionStore()
    assert store.get("nope") is None


def test_session_has_topomap():
    store = SessionStore()
    s = store.create(goal="g", goal_objects=[])
    assert s.topomap is not None
    assert s.topomap.graph.number_of_nodes() == 0
