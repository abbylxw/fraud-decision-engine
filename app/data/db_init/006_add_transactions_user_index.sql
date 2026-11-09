-- Purchase-history fallback index. When Redis has no history for a user, velocity.py's
-- _purchase_counts_from_db reads that user's last 90 days of rows by user_id and purchase_time.
-- Without this index every such read is a full scan of the audit table (~30 ms at 431k rows,
-- growing with the table); see docs/performance_2026-11.md.
-- Idempotent, so it is safe on a fresh database (runs after 005). On an existing database,
-- build it without blocking writes instead:
--   psql "$DATABASE_URL" -c "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_user_time ON transactions (user_id, purchase_time);"
CREATE INDEX IF NOT EXISTS idx_transactions_user_time ON transactions (user_id, purchase_time);
