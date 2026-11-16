-- Audit column for the model feature did_other_uid_all (added 2026-11-15): the number of
-- OTHER accounts ever seen on the transaction's device. It replaced did_txn_cnt_all as the
-- model input the day that one shipped (docs/feature_ablation.md section 8); the
-- did_txn_cnt_all column stays as an audited, monitored value.
-- Idempotent. On an existing database:
--   psql "$DATABASE_URL" -f app/data/db_init/009_add_device_other_accounts.sql
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS did_other_uid_all int NOT NULL DEFAULT 0;
