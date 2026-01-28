"""
Configuration Manager for Resume Skill Recognition System
Loads and manages system configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages system configuration and provides easy access to settings."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure one config instance."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        project_root = Path(__file__).parent.parent
        
        paths = self._config.get('paths', {})
        for key, path in paths.items():
            full_path = project_root / path
            full_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'paths.data_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """
        Get absolute path from config.
        
        Args:
            key: Key in paths configuration
            
        Returns:
            Absolute Path object
        """
        project_root = Path(__file__).parent.parent
        relative_path = self.get(f'paths.{key}')
        
        if relative_path:
            return project_root / relative_path
        else:
            raise ValueError(f"Path key '{key}' not found in configuration")
    
    @property
    def config(self) -> Dict:
        """Get entire configuration dictionary."""
        return self._config


# Global config instance
config = ConfigManager()
