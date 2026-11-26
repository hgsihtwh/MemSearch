"""Добавлены аккаунты и сессии; старые коллекции сохранены без назначения владельца.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-26 14:01:51.726685
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=True),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(length=60), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "email IS NULL OR (email = lower(btrim(email)) AND length(email) > 0)",
            name="ck_users_normalized_email",
        ),
        sa.CheckConstraint(
            "password_hash IS NULL OR email IS NOT NULL", name="ck_users_password_has_email"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("access_hash", sa.String(length=64), nullable=False),
        sa.Column("refresh_hash", sa.String(length=64), nullable=False),
        sa.Column("access_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refresh_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "refresh_expires_at > access_expires_at", name="ck_auth_sessions_expiry_order"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("access_hash"),
        sa.UniqueConstraint("refresh_hash"),
    )
    op.create_index(op.f("ix_auth_sessions_user_id"), "auth_sessions", ["user_id"], unique=False)
    op.create_table(
        "one_time_tokens",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("purpose", sa.String(length=16), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "purpose IN ('password_reset', 'email_setup')", name="ck_one_time_tokens_purpose"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(
        op.f("ix_one_time_tokens_user_id"), "one_time_tokens", ["user_id"], unique=False
    )
    op.create_table(
        "telegram_logins",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("poll_token_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'expired', 'consumed')",
            name="ck_telegram_logins_status",
        ),
        sa.CheckConstraint(
            "status NOT IN ('approved', 'consumed') OR user_id IS NOT NULL",
            name="ck_telegram_logins_confirmed_user",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("poll_token_hash"),
    )
    op.create_index(
        op.f("ix_telegram_logins_user_id"), "telegram_logins", ["user_id"], unique=False
    )
    op.create_table(
        "user_identities",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("provider", sa.String(length=16), nullable=False),
        sa.Column("provider_user_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.Text(), nullable=True),
        sa.CheckConstraint("provider = 'telegram'", name="ck_user_identities_provider"),
        sa.CheckConstraint("provider_user_id > 0", name="ck_user_identities_positive_id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "provider_user_id"),
        sa.UniqueConstraint("user_id", "provider"),
    )
    op.add_column("collections", sa.Column("user_id", sa.UUID(), nullable=True))
    op.alter_column("collections", "key_hash", existing_type=sa.VARCHAR(length=64), nullable=True)
    op.create_unique_constraint("collections_user_id_key", "collections", ["user_id"])
    op.create_foreign_key(
        "collections_user_id_fkey", "collections", "users", ["user_id"], ["id"], ondelete="CASCADE"
    )
    op.create_check_constraint(
        "ck_collections_owner_or_legacy_key",
        "collections",
        "(user_id IS NOT NULL AND key_hash IS NULL) OR (user_id IS NULL AND key_hash IS NOT NULL)",
    )
    op.add_column("telegram_links", sa.Column("username", sa.Text(), nullable=True))
    op.add_column("telegram_links", sa.Column("code_hash", sa.String(length=64), nullable=True))
    op.drop_constraint(op.f("telegram_links_link_code_key"), "telegram_links", type_="unique")
    op.create_unique_constraint("telegram_links_code_hash_key", "telegram_links", ["code_hash"])
    op.execute(
        "UPDATE telegram_links SET code_hash = encode(sha256(convert_to(link_code, 'UTF8')), 'hex') "
        "WHERE link_code IS NOT NULL"
    )
    op.drop_column("telegram_links", "link_code")


def downgrade() -> None:
    # Нельзя потерять аккаунты или активные попытки входа при откате схемы.
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM users) OR EXISTS (SELECT 1 FROM telegram_logins) THEN
                RAISE EXCEPTION 'Cannot downgrade 0002 while account data exists';
            END IF;
        END $$
    """)
    op.add_column("telegram_links", sa.Column("link_code", sa.String(32), nullable=True))
    # Исходные одноразовые коды из хешей не восстанавливаются; старые коды отзываются.
    op.execute("UPDATE telegram_links SET code_expires_at = NULL WHERE code_hash IS NOT NULL")
    op.drop_constraint("telegram_links_code_hash_key", "telegram_links", type_="unique")
    op.create_unique_constraint("telegram_links_link_code_key", "telegram_links", ["link_code"])
    op.drop_column("telegram_links", "code_hash")
    op.drop_column("telegram_links", "username")
    op.drop_constraint("ck_collections_owner_or_legacy_key", "collections", type_="check")
    op.drop_constraint("collections_user_id_fkey", "collections", type_="foreignkey")
    op.drop_constraint("collections_user_id_key", "collections", type_="unique")
    op.alter_column("collections", "key_hash", existing_type=sa.String(64), nullable=False)
    op.drop_column("collections", "user_id")
    op.drop_table("user_identities")
    op.drop_index("ix_telegram_logins_user_id", table_name="telegram_logins")
    op.drop_table("telegram_logins")
    op.drop_index("ix_one_time_tokens_user_id", table_name="one_time_tokens")
    op.drop_table("one_time_tokens")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("users")
