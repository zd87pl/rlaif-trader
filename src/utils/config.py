"""Configuration management"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Main configuration class using Pydantic settings"""

    # API Keys
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    alpaca_api_key: str = Field(default="", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL"
    )

    # Device
    device: str = Field(default="cuda", env="DEVICE")

    # Paths
    data_dir: Path = Field(default=Path("./historical_data"))
    log_dir: Path = Field(default=Path("./logs"))
    model_dir: Path = Field(default=Path("./models/checkpoints"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
_config: Optional[Dict[str, Any]] = None
_settings: Optional[Config] = None


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    global _config

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    return _config


def get_config() -> Dict[str, Any]:
    """Get the global configuration"""
    global _config

    if _config is None:
        _config = load_config()

    return _config


def get_settings() -> Config:
    """Get Pydantic settings instance"""
    global _settings

    if _settings is None:
        _settings = Config()

    return _settings


def get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "rl.td3.learning_rate")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
