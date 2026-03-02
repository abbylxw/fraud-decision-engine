from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    database_url: str = "postgresql://postgres:postgres@localhost:5432/fraud"

    model_config = {"env_file": ".env"}


settings = Settings()
