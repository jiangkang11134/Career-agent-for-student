"""Account and session authentication for the web app."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import Boolean, Column, DateTime, MetaData, String, Table, create_engine, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

ACCOUNT_RE = re.compile(r"^\S{10,20}$")
DEFAULT_SESSION_DAYS = 30

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("user_id", String(32), primary_key=True),
    # The physical column keeps the old name for zero-downtime compatibility.
    Column("email", String(255), nullable=False, unique=True, index=True),
    Column("display_name", String(120), nullable=False),
    Column("password_hash", String(255), nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

auth_sessions = Table(
    "auth_sessions",
    metadata,
    Column("token_hash", String(64), primary_key=True),
    Column("user_id", String(32), nullable=False, index=True),
    Column("created_at", DateTime, nullable=False),
    Column("expires_at", DateTime, nullable=False, index=True),
    Column("last_seen_at", DateTime, nullable=False),
    Column("revoked", Boolean, nullable=False, default=False),
)


@dataclass(frozen=True, slots=True)
class AuthUser:
    user_id: str
    account: str
    display_name: str

    @property
    def email(self) -> str:
        return self.account

    @property
    def tenant_id(self) -> str:
        return self.user_id

    def to_dict(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "account": self.account,
            "display_name": self.display_name,
        }


class AuthError(ValueError):
    pass


def default_database_url() -> str:
    explicit = str(os.environ.get("SMART_AGENT_DATABASE_URL") or "").strip()
    if explicit:
        return explicit
    host = str(os.environ.get("SMART_AGENT_MYSQL_HOST") or "127.0.0.1").strip()
    port = str(os.environ.get("SMART_AGENT_MYSQL_PORT") or "3306").strip()
    database = str(os.environ.get("SMART_AGENT_MYSQL_DATABASE") or "smart_agent").strip()
    user = str(os.environ.get("SMART_AGENT_MYSQL_USER") or "smart_agent").strip()
    password = str(os.environ.get("SMART_AGENT_MYSQL_PASSWORD") or "smart-agent-dev")
    return (
        "mysql+pymysql://"
        + quote_plus(user)
        + ":"
        + quote_plus(password)
        + "@"
        + host
        + ":"
        + port
        + "/"
        + quote_plus(database)
        + "?charset=utf8mb4"
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_account(account: str) -> str:
    value = str(account or "").strip().casefold()
    if not ACCOUNT_RE.fullmatch(value):
        raise AuthError("account must be 10-20 non-whitespace characters")
    return value


def _hash_password(password: str, *, salt: str | None = None) -> str:
    if len(password) < 8:
        raise AuthError("password must be at least 8 characters")
    selected_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(selected_salt),
        240_000,
    )
    return "pbkdf2_sha256$240000$" + selected_salt + "$" + digest.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), expected)


def _token_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


class AuthService:
    def __init__(self, engine: Engine | None = None, *, session_days: int = DEFAULT_SESSION_DAYS) -> None:
        self.engine = engine or create_engine(default_database_url(), pool_pre_ping=True, future=True)
        self.session_days = max(1, int(session_days))

    @classmethod
    def from_env(cls) -> "AuthService":
        return cls()

    def init_schema(self) -> None:
        metadata.create_all(self.engine)

    def register(self, account: str, password: str, display_name: str = "") -> tuple[AuthUser, str]:
        normalized_account = _normalize_account(account)
        now = _utc_now()
        user = AuthUser(
            user_id=uuid.uuid4().hex,
            account=normalized_account,
            display_name=str(display_name or "").strip() or normalized_account,
        )
        with self.engine.begin() as conn:
            self.init_schema()
            try:
                conn.execute(
                    users.insert().values(
                        user_id=user.user_id,
                        email=user.account,
                        display_name=user.display_name,
                        password_hash=_hash_password(password),
                        created_at=now,
                        updated_at=now,
                    )
                )
            except IntegrityError as exc:
                raise AuthError("account already registered") from exc
        return user, self.create_session(user.user_id)

    def authenticate(self, account: str, password: str) -> tuple[AuthUser, str]:
        normalized_account = _normalize_account(account)
        self.init_schema()
        with self.engine.begin() as conn:
            row = conn.execute(select(users).where(users.c.email == normalized_account)).mappings().first()
            if row is None or not _verify_password(password, str(row["password_hash"])):
                raise AuthError("invalid account or password")
            user = AuthUser(
                user_id=str(row["user_id"]),
                account=str(row["email"]),
                display_name=str(row["display_name"]),
            )
        return user, self.create_session(user.user_id)

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = _utc_now()
        with self.engine.begin() as conn:
            self.init_schema()
            conn.execute(
                auth_sessions.insert().values(
                    token_hash=_token_hash(token),
                    user_id=str(user_id),
                    created_at=now,
                    expires_at=now + timedelta(days=self.session_days),
                    last_seen_at=now,
                    revoked=False,
                )
            )
        return token

    def user_from_token(self, token: str) -> AuthUser | None:
        if not token:
            return None
        now = _utc_now()
        self.init_schema()
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(
                        auth_sessions.c.token_hash,
                        auth_sessions.c.expires_at,
                        auth_sessions.c.revoked,
                        users.c.user_id,
                        users.c.email,
                        users.c.display_name,
                    )
                    .select_from(auth_sessions.join(users, auth_sessions.c.user_id == users.c.user_id))
                    .where(auth_sessions.c.token_hash == _token_hash(token))
                )
                .mappings()
                .first()
            )
            if row is None or bool(row["revoked"]) or row["expires_at"] <= now:
                return None
            conn.execute(
                update(auth_sessions)
                .where(auth_sessions.c.token_hash == row["token_hash"])
                .values(last_seen_at=now)
            )
            return AuthUser(
                user_id=str(row["user_id"]),
                account=str(row["email"]),
                display_name=str(row["display_name"]),
            )

    def revoke(self, token: str) -> None:
        if not token:
            return
        self.init_schema()
        with self.engine.begin() as conn:
            conn.execute(
                update(auth_sessions)
                .where(auth_sessions.c.token_hash == _token_hash(token))
                .values(revoked=True, last_seen_at=_utc_now())
            )


def auth_health() -> dict[str, Any]:
    service = AuthService.from_env()
    service.init_schema()
    with service.engine.begin() as conn:
        conn.execute(select(users.c.user_id).limit(1)).first()
    return {"ok": True, "database": service.engine.url.get_backend_name()}
