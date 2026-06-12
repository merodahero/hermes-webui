"""Regression tests for #3994 / #3985 — _get_or_materialize_session().

rename / move / update of a CLI/agent session that isn't yet in the WebUI store
should materialize it from CLI metadata (mirroring /api/session/archive) instead
of 404ing — while still refusing to mutate a read-only (messaging / Claude Code)
session.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_materialize_returns_in_store_session_directly():
    """When get_session() succeeds, the helper returns it (after full-load) untouched."""
    import api.routes as routes

    existing = SimpleNamespace(session_id="s1", profile="default", messages=[])
    with patch("api.routes.get_session", return_value=existing), \
         patch("api.routes._ensure_full_session_before_mutation", return_value=existing):
        out = routes._get_or_materialize_session("s1")
    assert out is existing


def test_materialize_missing_everywhere_raises_keyerror():
    """No WebUI session and no CLI metadata → KeyError (caller maps to 404)."""
    import api.routes as routes

    with patch("api.routes.get_session", side_effect=KeyError("s1")), \
         patch("api.routes._lookup_cli_session_metadata", return_value={}):
        with pytest.raises(KeyError):
            routes._get_or_materialize_session("s1")


def test_materialize_readonly_session_raises_permissionerror():
    """A read-only imported session (messaging / Claude Code) must not be materialized
    for mutation — the helper raises PermissionError (caller maps to 403)."""
    import api.routes as routes

    with patch("api.routes.get_session", side_effect=KeyError("ro1")), \
         patch("api.routes._lookup_cli_session_metadata", return_value={"read_only": True, "source_tag": "claude_code"}):
        with pytest.raises(PermissionError):
            routes._get_or_materialize_session("ro1")


def test_materialize_cli_session_imports_full_history():
    """A regular (non-messaging, non-read-only) CLI session is materialized via
    import_cli_session with its message history."""
    import api.routes as routes

    cli_meta = {
        "read_only": False,
        "title": "CLI chat",
        "model": "gpt-test",
        "profile": "default",
        "source_tag": "cli",
    }
    imported = SimpleNamespace(session_id="cli1", profile="default", messages=[{"role": "user", "content": "hi"}])
    with patch("api.routes.get_session", side_effect=KeyError("cli1")), \
         patch("api.routes._lookup_cli_session_metadata", return_value=cli_meta), \
         patch("api.routes._is_messaging_session_record", return_value=False), \
         patch("api.routes.get_cli_session_messages", return_value=[{"role": "user", "content": "hi"}]), \
         patch("api.routes.title_from", return_value="CLI chat"), \
         patch("api.routes.import_cli_session", return_value=imported) as mock_import:
        out = routes._get_or_materialize_session("cli1")
    assert out is imported
    assert mock_import.called
    # source metadata is stamped onto the materialized session
    assert getattr(out, "is_cli_session", None) is True


def test_materialize_rejects_stored_readonly_session():
    """An already-STORED read-only session must be refused on the happy path too —
    get_session() succeeding doesn't make it mutable (Codex CORE #1)."""
    import api.routes as routes

    ro = SimpleNamespace(session_id="ro_stored", profile="default", messages=[], read_only=True)
    with patch("api.routes.get_session", return_value=ro), \
         patch("api.routes._ensure_full_session_before_mutation", return_value=ro):
        with pytest.raises(PermissionError):
            routes._get_or_materialize_session("ro_stored")


def test_materialize_allows_stored_messaging_session_without_readonly():
    """A stored messaging session that ALREADY owns its sidecar is mutable on the
    happy path (the messaging-fork concern only applies to the materialize
    fallback that would CREATE a sidecar). Only an explicit read_only flag blocks
    a stored session."""
    import api.routes as routes

    msg = SimpleNamespace(session_id="msg_stored", profile="default", messages=[],
                          session_source="messaging", source_tag="telegram", read_only=False)
    with patch("api.routes.get_session", return_value=msg), \
         patch("api.routes._ensure_full_session_before_mutation", return_value=msg):
        out = routes._get_or_materialize_session("msg_stored")
    assert out is msg


def test_materialize_rejects_messaging_cli_meta_without_readonly_flag():
    """Messaging cli_meta lacking an explicit read_only flag must still be refused
    — agent rows normalize messaging sources without setting read_only, and
    state.db is the source of truth, so a writable sidecar would fork it (Codex CORE #2)."""
    import api.routes as routes

    cli_meta = {"title": "tg chat", "model": "gpt-test", "source_tag": "telegram", "session_source": "messaging"}
    with patch("api.routes.get_session", side_effect=KeyError("msg1")), \
         patch("api.routes._lookup_cli_session_metadata", return_value=cli_meta), \
         patch("api.routes.import_cli_session") as mock_import, \
         patch("api.routes.Session") as MockSession:
        with pytest.raises(PermissionError):
            routes._get_or_materialize_session("msg1")
    assert not mock_import.called, "must not import a messaging session"
    assert not MockSession.called, "must not build a writable messaging stub"


def test_materialize_allows_cron_cli_session():
    """A cron session (state.db source='cron') must be allowed through the
    messaging denylist and materialized normally.

    Background (issue #3975, maintainer verification ask): cron-job sessions
    live in state.db but typically don't appear in ``get_cli_sessions()``, so
    they rely on the materialize fallback.  The maintainer wanted explicit
    confirmation that cron-origin sessions are not refused by the messaging /
    claude_code / external_agent / gateway denylist — they should be
    claimable since they're state.db-backed agent sessions.  This test pins
    that contract: a cron-shaped cli_meta (source_tag='cron',
    session_source='other') is materialized via ``import_cli_session`` like
    any other non-messaging, non-readonly CLI session."""
    import api.routes as routes

    cli_meta = {
        "read_only": False,
        "title": "Scheduled run",
        "model": "gpt-test",
        "profile": "default",
        "source_tag": "cron",
        "session_source": "other",
    }
    imported = SimpleNamespace(
        session_id="cron1", profile="default",
        messages=[{"role": "user", "content": "scheduled task"}],
    )
    with patch("api.routes.get_session", side_effect=KeyError("cron1")), \
         patch("api.routes._lookup_cli_session_metadata", return_value=cli_meta), \
         patch("api.routes._is_messaging_session_record", return_value=False), \
         patch("api.routes.get_cli_session_messages",
               return_value=[{"role": "user", "content": "scheduled task"}]), \
         patch("api.routes.title_from", return_value="Scheduled run"), \
         patch("api.routes.import_cli_session", return_value=imported) as mock_import:
        out = routes._get_or_materialize_session("cron1")
    assert out is imported
    assert mock_import.called, "cron session must be materialized via import_cli_session"
    # cron is claimable: no PermissionError raised, no messaging rejection
    assert getattr(out, "is_cli_session", None) is True
    assert getattr(out, "source_tag", None) == "cron"


def test_materialize_rejects_cron_with_explicit_readonly_flag():
    """Cron + read_only=True must still be refused — explicit read_only wins
    over source classification (mirrors the messaging + read_only contract
    in ``_is_messaging_session_record``)."""
    import api.routes as routes

    cli_meta = {
        "read_only": True,
        "title": "Locked cron",
        "model": "gpt-test",
        "source_tag": "cron",
        "session_source": "other",
    }
    with patch("api.routes.get_session", side_effect=KeyError("cron_ro")), \
         patch("api.routes._lookup_cli_session_metadata", return_value=cli_meta), \
         patch("api.routes._is_messaging_session_record", return_value=False), \
         patch("api.routes.import_cli_session") as mock_import:
        with pytest.raises(PermissionError):
            routes._get_or_materialize_session("cron_ro")
    assert not mock_import.called, "read-only cron must not be imported"


def test_materialize_falls_back_to_state_db_when_no_cli_meta(monkeypatch):
    """State.db-only sessions (cron, TUI past CLI cap, etc.) — the actual
    user-pain path from #3975 and PR #3901's 171-message TUI session.

    When ``_lookup_cli_session_metadata`` returns empty but ``state_db_has_session``
    returns True with recoverable messages, the helper synthesizes a cli_meta
    from the state.db row and materializes via ``import_cli_session`` with
    state.db messages as the source.
    """
    import api.routes as routes
    from types import SimpleNamespace

    synthetic_meta = {
        "title": "Scheduled run",
        "model": "gpt-test",
        "source_tag": "cron",
        "raw_source": "cron",
        "session_source": "cron",
        "profile": "default",
        "created_at": 1718000000.0,
        "updated_at": 1718000060.0,
        "_state_db_synthesized": True,
    }
    state_msgs = [{"role": "user", "content": "scheduled task output"}]
    imported = SimpleNamespace(
        session_id="cron_abc_12345", profile="default", messages=state_msgs,
    )
    with patch("api.routes.get_session", side_effect=KeyError("cron_abc_12345")), \
         patch("api.routes._lookup_cli_session_metadata", return_value={}), \
         patch("api.routes._state_db_synthetic_cli_meta", return_value=synthetic_meta), \
         patch("api.routes._state_db_synthetic_messages", return_value=state_msgs), \
         patch("api.routes._is_messaging_session_record", return_value=False), \
         patch("api.routes.import_cli_session", return_value=imported) as mock_import, \
         patch("api.routes.get_cli_session_messages", return_value=[]) as mock_cli_msgs:
        out = routes._get_or_materialize_session("cron_abc_12345")
    assert out is imported
    assert mock_import.called
    # Critical: state.db messages path was used, NOT the CLI messages path
    assert not mock_cli_msgs.called, "must use state.db messages for state.db-only sessions"
    assert getattr(out, "is_cli_session", None) is True
    assert getattr(out, "source_tag", None) == "cron"


def test_materialize_state_db_no_messages_raises_keyerror(monkeypatch):
    """State.db session with no recoverable messages must 404 (KeyError),
    not silently build an empty stub.  The cron user should not see an
    empty session in the WebUI."""
    import api.routes as routes
    with patch("api.routes.get_session", side_effect=KeyError("cron_empty")), \
         patch("api.routes._lookup_cli_session_metadata", return_value={}), \
         patch("api.routes._state_db_synthetic_cli_meta", return_value={}), \
         patch("api.routes.import_cli_session") as mock_import:
        with pytest.raises(KeyError):
            routes._get_or_materialize_session("cron_empty")
    assert not mock_import.called


def test_materialize_state_db_synthetic_helper_returns_empty_for_messaging():
    """The state.db synthetic helper must return an empty dict (refusing to
    build a cli_meta) when the state.db session's source is a known messaging
    surface — those should still flow through the existing messaging-rejection
    gate, not the state.db fallback path."""
    from api.routes import _state_db_synthetic_cli_meta

    # Simulate a state.db row with a messaging source
    fake_row = {
        "id": "tg_abc_12345",
        "title": "Telegram chat",
        "model": "gpt-test",
        "source": "telegram",
        "started_at": 1718000000.0,
        "ended_at": 1718000060.0,
        "profile": "default",
        "workspace": None,
        "last_activity": 1718000060.0,
    }

    with patch("api.routes._active_state_db_path") as mock_path, \
         patch("api.routes.get_state_db_session_summary",
               return_value={"message_count": 5, "last_message_at": 1718000060.0}), \
         patch("sqlite3.connect") as mock_connect:
        mock_path.return_value.exists.return_value = True
        # Build a fake connection that returns the row
        import sqlite3 as sqlite3_mod
        fake_conn = mock_connect.return_value.__enter__.return_value
        fake_conn.execute.return_value.fetchone.return_value = fake_row
        # But we still need PRAGMA to return id column
        fake_conn.execute.return_value.fetchall.return_value = [("id",), ("title",),
                                                                 ("model",), ("source",),
                                                                 ("started_at",), ("ended_at",),
                                                                 ("profile",),
                                                                 ("workspace",),
                                                                 ("last_activity",)]
        # And the SELECT row fetchone returns the row
        # Need a sequence of returns for the execute calls
        call_count = {"n": 0}

        class FakeCursor:
            def execute(self, *a, **kw):
                pass
            def fetchall(self):
                return [("id",), ("title",), ("model",), ("source",),
                        ("started_at",), ("ended_at",), ("profile",),
                        ("workspace",), ("last_activity",)]
            def fetchone(self):
                return fake_row
        fake_conn.execute = lambda *a, **kw: FakeCursor()

        result = _state_db_synthetic_cli_meta("tg_abc_12345")

    # Should refuse to synthesize for messaging sources
    assert result == {}, f"state.db helper must refuse messaging sources, got {result}"
