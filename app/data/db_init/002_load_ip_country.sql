CREATE TEMP TABLE ip_country_stage (
    ip_from_text text,
    ip_to_text   text,
    country      text
);

COPY ip_country_stage(ip_from_text, ip_to_text, country)
FROM '/data/IpAddress_to_Country.csv'
WITH (FORMAT csv, HEADER true);

INSERT INTO ip_country (ip_from, ip_to, country)
SELECT
  (ip_from_text::numeric)::bigint,
  (ip_to_text::numeric)::bigint,
  country
FROM ip_country_stage;

DROP TABLE ip_country_stage;
