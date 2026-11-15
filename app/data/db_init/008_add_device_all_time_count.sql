-- All-time device transaction count (model feature did_txn_cnt_all, added 2026-11-15).
-- 1. The audit column that stores the feature value used for each decision.
-- 2. The index behind velocity.py's _device_txn_count_from_db, which runs when Redis has no
--    did:txn_all counter for a device (a new device, or Redis lost the key). Without it that
--    lookup is a full scan of the audit table on most requests.
-- Idempotent, so it is safe on a fresh database (runs after 007). On an existing database:
--   psql "$DATABASE_URL" -c "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS did_txn_cnt_all int NOT NULL DEFAULT 0;"
--   psql "$DATABASE_URL" -c "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_device_time ON transactions (device_id, purchase_time);"
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS did_txn_cnt_all int NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_transactions_device_time ON transactions (device_id, purchase_time);
