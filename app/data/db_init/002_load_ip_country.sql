COPY ip_country(ip_from, ip_to, country)
FROM '/data/IpAddress_to_Country.csv'
WITH (FORMAT csv, HEADER true);
