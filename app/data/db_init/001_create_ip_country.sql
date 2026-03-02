CREATE TABLE IF NOT EXISTS ip_country (
  id bigserial PRIMARY KEY,
  ip_from bigint NOT NULL,
  ip_to   bigint NOT NULL,
  country text   NOT NULL,
  ip_range int8range GENERATED ALWAYS AS (int8range(ip_from, ip_to, '[]')) STORED
);

CREATE INDEX IF NOT EXISTS ix_ip_country_range_gist
ON ip_country USING GIST (ip_range);