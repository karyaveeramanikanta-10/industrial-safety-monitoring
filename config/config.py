"""
Configuration management for Industrial Safety Monitoring System.

Loads settings from YAML configuration file and provides typed access
to all configuration parameters through a singleton pattern.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger('safety_monitor')


@dataclass
class DetectionConfig:
    """Person detection configuration."""
    confidence_threshold: float = 0.5
    model_path: str = 'models/person_detector'
    input_size: int = 320
    max_detections: int = 20
    nms_threshold: float = 0.4


@dataclass
class PPEConfig:
    """PPE detection configuration."""
    required_items: List[str] = field(default_factory=lambda: ['helmet', 'vest'])
    color_ranges: Dict[str, Dict] = field(default_factory=dict)
    min_area_ratios: Dict[str, float] = field(default_factory=lambda: {
        'helmet': 0.12,
        'vest': 0.18,
        'mask': 0.10,
        'goggles': 0.05,
        'gloves': 0.08,
        'ear_protection': 0.05,
        'shoes': 0.10,
    })


@dataclass
class TrackerConfig:
    """Object tracker configuration."""
    algorithm: str = 'centroid'  # 'centroid' or 'sort'
    max_disappeared: int = 50
    max_distance: int = 50


@dataclass
class AlertConfig:
    """Alert system configuration."""
    cooldown_seconds: int = 30
    sound_enabled: bool = True
    sound_file: str = 'assets/alert_sound.wav'
    email_enabled: bool = False
    smtp_server: str = 'smtp.gmail.com'
    smtp_port: int = 587
    smtp_username: str = ''
    smtp_password: str = ''
    email_recipients: List[str] = field(default_factory=list)
    sms_enabled: bool = False
    twilio_account_sid: str = ''
    twilio_auth_token: str = ''
    twilio_from_number: str = ''
    sms_recipients: List[str] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    """Database configuration (MongoDB Atlas)."""
    mongo_uri: str = ''  # Loaded from MONGO_URI env var
    db_name: str = 'safety_monitoring'
    db_path: str = ''  # Legacy compat — unused with MongoDB



@dataclass
class VideoConfig:
    """Video capture configuration."""
    source: int = 0
    width: int = 640
    height: int = 480
    fps_cap: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = 'INFO'
    log_file: str = 'logs/system.log'
    max_file_size_mb: int = 5
    backup_count: int = 5


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str = 'Industrial Safety Monitoring System'
    refresh_rate: int = 1
    max_alerts_display: int = 20
    chart_history_hours: int = 24


# Default PPE color ranges in HSV
DEFAULT_PPE_COLOR_RANGES = {
    'helmet': {
        'white':  {'lower': [0, 0, 180],   'upper': [180, 30, 255]},
        'yellow': {'lower': [20, 80, 80],  'upper': [35, 255, 255]},
        'blue':   {'lower': [100, 80, 80], 'upper': [130, 255, 255]},
        'red':    {'lower': [0, 100, 100], 'upper': [10, 255, 255]},
        'orange': {'lower': [10, 100, 100], 'upper': [20, 255, 255]},
    },
    'vest': {
        'hi_vis_yellow': {'lower': [20, 100, 100], 'upper': [35, 255, 255]},
        'hi_vis_orange': {'lower': [5, 100, 100],  'upper': [18, 255, 255]},
    },
    'gloves': {
        'blue':   {'lower': [100, 50, 50],  'upper': [130, 255, 255]},
        'orange': {'lower': [5, 100, 100],  'upper': [20, 255, 255]},
    },
    'mask': {
        'white': {'lower': [0, 0, 180],   'upper': [180, 30, 255]},
        'blue':  {'lower': [90, 50, 50],  'upper': [130, 255, 255]},
    },
    'goggles': {
        'clear': {'lower': [0, 0, 200], 'upper': [180, 30, 255]},
    },
    'ear_protection': {
        'yellow': {'lower': [20, 80, 80],  'upper': [35, 255, 255]},
        'red':    {'lower': [0, 100, 100], 'upper': [10, 255, 255]},
    },
    'shoes': {
        'black': {'lower': [0, 0, 0],    'upper': [180, 255, 50]},
        'brown': {'lower': [10, 50, 20], 'upper': [20, 200, 150]},
    },
}


class Config:
    """Central configuration manager (Singleton).

    Loads settings from a YAML file and provides typed access through
    dataclass-based sub-configurations.

    Usage:
        config = Config.get_instance()
        threshold = config.detection.confidence_threshold
    """

    _instance: Optional['Config'] = None

    def __init__(self, config_path: str = 'config/settings.yaml'):
        self.config_path = config_path
        self._raw: Dict[str, Any] = {}

        # Initialize sub-configs with defaults
        self.detection = DetectionConfig()
        self.ppe = PPEConfig()
        self.tracker = TrackerConfig()
        self.alerts = AlertConfig()
        self.database = DatabaseConfig()
        self.video = VideoConfig()
        self.logging = LoggingConfig()
        self.dashboard = DashboardConfig()

        # Load from file
        self._load(config_path)

    @classmethod
    def get_instance(cls, config_path: str = None) -> 'Config':
        """Get or create singleton Config instance."""
        if cls._instance is None:
            cls._instance = cls(config_path or 'config/settings.yaml')
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        cls._instance = None

    def _load(self, config_path: str):
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            self.ppe.color_ranges = DEFAULT_PPE_COLOR_RANGES
            return

        try:
            with open(config_path, 'r') as f:
                self._raw = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
            self._raw = {}

        # Map YAML sections to dataclass configs
        self._apply_section('detection', self.detection)
        self._apply_section('ppe', self.ppe)
        self._apply_section('tracker', self.tracker)
        self._apply_section('alerts', self.alerts)
        self._apply_section('database', self.database)
        self._apply_section('video', self.video)
        self._apply_section('logging', self.logging)
        self._apply_section('dashboard', self.dashboard)

        # Set default color ranges if not specified
        if not self.ppe.color_ranges:
            self.ppe.color_ranges = DEFAULT_PPE_COLOR_RANGES

    def _apply_section(self, section_name: str, target):
        """Apply YAML section values to a dataclass instance."""
        section = self._raw.get(section_name, {})
        if not isinstance(section, dict):
            return
        for key, value in section.items():
            if hasattr(target, key):
                setattr(target, key, value)

    def save(self, path: str = None):
        """Save current configuration to YAML file."""
        path = path or self.config_path
        data = {
            'detection': self._dataclass_to_dict(self.detection),
            'ppe': self._dataclass_to_dict(self.ppe),
            'tracker': self._dataclass_to_dict(self.tracker),
            'alerts': self._dataclass_to_dict(self.alerts),
            'database': self._dataclass_to_dict(self.database),
            'video': self._dataclass_to_dict(self.video),
            'logging': self._dataclass_to_dict(self.logging),
            'dashboard': self._dataclass_to_dict(self.dashboard),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {path}")

    @staticmethod
    def _dataclass_to_dict(obj) -> dict:
        """Convert dataclass to dict for YAML serialization."""
        result = {}
        for key in obj.__dataclass_fields__:
            value = getattr(obj, key)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            result[key] = value
        return result

    def __repr__(self):
        return (
            f"Config(\n"
            f"  detection={self.detection},\n"
            f"  ppe={self.ppe},\n"
            f"  tracker={self.tracker},\n"
            f"  alerts={self.alerts},\n"
            f"  database={self.database},\n"
            f"  video={self.video},\n"
            f"  logging={self.logging},\n"
            f"  dashboard={self.dashboard}\n"
            f")"
        )
