-- Audit columns: which model made each decision, its raw score, and which rule denied.
-- model_score is NULL when a hard rule denied before the model ran.
-- Idempotent, so it is safe on a fresh database (runs after 004) and can be applied by hand
-- to an existing one: psql "$DATABASE_URL" -f app/data/db_init/005_add_model_audit_columns.sql
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS model_version text;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS model_score   double precision;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS decision_rule text;
CREATE INDEX IF NOT EXISTS idx_transactions_model_version ON transactions (model_version);
