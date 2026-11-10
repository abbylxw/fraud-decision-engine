-- country_txn_count_{5min,10min,1h} were computed and stored but used by nothing: not among
-- the model's inputs (model_features.json) and not in any hard, soft or shadow rule. Dropped
-- 2026-11-09 along with the 3 Redis ZCOUNTs per request that fed them; 24h/7d/30d stay (model
-- inputs). 004 no longer creates these columns, so on a fresh database this is a no-op.
-- Idempotent; apply by hand to an existing database:
--   psql "$DATABASE_URL" -f app/data/db_init/007_drop_unused_country_windows.sql
ALTER TABLE transactions DROP COLUMN IF EXISTS country_txn_count_5min;
ALTER TABLE transactions DROP COLUMN IF EXISTS country_txn_count_10min;
ALTER TABLE transactions DROP COLUMN IF EXISTS country_txn_count_1h;
