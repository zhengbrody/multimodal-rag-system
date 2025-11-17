"""
Configuration utilities for loading and managing settings
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('models.embedding.model_name')
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment"""
        return os.getenv('OPENAI_API_KEY', '')

    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key from environment"""
        return os.getenv('ANTHROPIC_API_KEY', '')

    @property
    def pinecone_api_key(self) -> str:
        """Get Pinecone API key from environment"""
        return os.getenv('PINECONE_API_KEY', '')


# Global config instance
config = Config()
