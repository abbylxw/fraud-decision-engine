CREATE TABLE IF NOT EXISTS device_data (
    device_id    text NOT NULL,
    user_id      text NOT NULL,
    did_uid_date date NOT NULL,
    PRIMARY KEY (device_id, user_id)
);
