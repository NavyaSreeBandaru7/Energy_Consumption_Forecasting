import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    name: str = os.getenv('DB_NAME', 'energy_forecasting')
    user: str = os.getenv('DB_USER', 'energy_user')
    password: str = os.getenv('DB_PASSWORD', '')
    driver: str = os.getenv('DB_DRIVER', 'postgresql')
    
    @property
    def connection_string(self) -> str:
        return f"{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class APIConfig:
    """API configuration settings"""
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    huggingface_token: str = os.getenv('HUGGINGFACE_TOKEN', '')
    weather_api_key: str = os.getenv('WEATHER_API_KEY', '')
    electricity_api_key: str = os.getenv('ELECTRICITY_API_KEY', '')
    
    # Rate limiting
    rate_limit_requests_per_minute: int = int(os.getenv('RATE_LIMIT_RPM', '60'))
    rate_limit_tokens_per_minute: int = int(os.getenv('RATE_LIMIT_TPM', '10000'))
    
    # Timeout settings
    api_timeout: int = int(os.getenv('API_TIMEOUT', '30'))
    max_retries: int = int(os.getenv('MAX_RETRIES', '3'))

@dataclass
class ModelConfig:
    """Machine Learning model configuration"""
    # Model paths
    model_save_path: str = os.getenv('MODEL_SAVE_PATH', 'models/')
    model_registry_path: str = os.getenv('MODEL_REGISTRY_PATH', 'registry/')
    
    # Training parameters
    train_test_split: float = float(os.getenv('TRAIN_TEST_SPLIT', '0.8'))
    validation_split: float = float(os.getenv('VALIDATION_SPLIT', '0.2'))
    random_state: int = int(os.getenv('RANDOM_STATE', '42'))
    
    # LSTM/Neural Network parameters
    lstm_units: int = int(os.getenv('LSTM_UNITS', '128'))
    dropout_rate: float = float(os.getenv('DROPOUT_RATE', '0.2'))
    learning_rate: float = float(os.getenv('LEARNING_RATE', '0.001'))
    batch_size: int = int(os.getenv('BATCH_SIZE', '64'))
    epochs: int = int(os.getenv('EPOCHS', '100'))
    early_stopping_patience: int = int(os.getenv('EARLY_STOPPING_PATIENCE', '20'))
    
    # Sequence parameters
    sequence_length: int = int(os.getenv('SEQUENCE_LENGTH', '24'))
    forecast_horizon: int = int(os.getenv('FORECAST_HORIZON', '168'))  # 7 days in hours
    
    # Ensemble parameters
    ensemble_models: List[str] = field(default_factory=lambda: [
        'random_forest', 'gradient_boosting', 'lstm', 'prophet'
    ])
    
    # Feature engineering
    lag_features: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 24, 168])
    
    # Prophet specific
    prophet_seasonality_mode: str = os.getenv('PROPHET_SEASONALITY_MODE', 'multiplicative')
    prophet_changepoint_prior_scale: float = float(os.getenv('PROPHET_CHANGEPOINT_PRIOR_SCALE', '0.05'))
    prophet_seasonality_prior_scale: float = float(os.getenv('PROPHET_SEASONALITY_PRIOR_SCALE', '0.1'))

@dataclass
class DataConfig:
    """Data processing configuration"""
    # Data paths
    data_path: str = os.getenv('DATA_PATH', 'data/')
    raw_data_path: str = os.getenv('RAW_DATA_PATH', 'data/raw/')
    processed_data_path: str = os.getenv('PROCESSED_DATA_PATH', 'data/processed/')
    cache_path: str = os.getenv('CACHE_PATH', 'cache/')
    
    # Data validation
    max_missing_percentage: float = float(os.getenv('MAX_MISSING_PERCENTAGE', '0.1'))
    outlier_detection_method: str = os.getenv('OUTLIER_DETECTION_METHOD', 'iqr')
    outlier_threshold: float = float(os.getenv('OUTLIER_THRESHOLD', '3.0'))
    
    # Data generation
    synthetic_data_periods: int = int(os.getenv('SYNTHETIC_DATA_PERIODS', '8760'))
    data_frequency: str = os.getenv('DATA_FREQUENCY', 'H')  # Hourly
    
    # Feature engineering
    enable_weather_features: bool = os.getenv('ENABLE_WEATHER_FEATURES', 'true').lower() == 'true'
    enable_calendar_features: bool = os.getenv('ENABLE_CALENDAR_FEATURES', 'true').lower() == 'true'
    enable_lag_features: bool = os.getenv('ENABLE_LAG_FEATURES', 'true').lower() == 'true'

@dataclass
class BusinessConfig:
    """Business logic configuration"""
    # Energy pricing
    energy_cost_per_kwh: float = float(os.getenv('ENERGY_COST_PER_KWH', '0.12'))
    peak_hour_multiplier: float = float(os.getenv('PEAK_HOUR_MULTIPLIER', '1.5'))
    off_peak_multiplier: float = float(os.getenv('OFF_PEAK_MULTIPLIER', '0.7'))
    
    # Peak hours definition
    peak_start_hour: int = int(os.getenv('PEAK_START_HOUR', '17'))
    peak_end_hour: int = int(os.getenv('PEAK_END_HOUR', '20'))
    off_peak_start_hour: int = int(os.getenv('OFF_PEAK_START_HOUR', '22'))
    off_peak_end_hour: int = int(os.getenv('OFF_PEAK_END_HOUR', '6'))
    
    # Thresholds
    peak_consumption_threshold: float = float(os.getenv('PEAK_CONSUMPTION_THRESHOLD', '0.8'))
    efficiency_target: float = float(os.getenv('EFFICIENCY_TARGET', '0.15'))  # 15% improvement
    
    # Alert thresholds
    high_consumption_alert_threshold: float = float(os.getenv('HIGH_CONSUMPTION_ALERT_THRESHOLD', '1.2'))
    anomaly_detection_threshold: float = float(os.getenv('ANOMALY_DETECTION_THRESHOLD', '2.0'))
    
    # Sustainability metrics
    carbon_intensity_kg_per_kwh: float = float(os.getenv('CARBON_INTENSITY', '0.4'))
    renewable_energy_percentage: float = float(os.getenv('RENEWABLE_PERCENTAGE', '0.3'))

@dataclass
class DeploymentConfig:
    """Deployment and infrastructure configuration"""
    # Environment
    environment: str = os.getenv('ENVIRONMENT', 'development')
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # Streamlit configuration
    streamlit_port: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    streamlit_host: str = os.getenv('STREAMLIT_HOST', '0.0.0.0')
    
    # API configuration
    api_port: int = int(os.getenv('API_PORT', '8000'))
    api_host: str = os.getenv('API_HOST', '0.0.0.0')
    api_workers: int = int(os.getenv('API_WORKERS', '4'))
    
    # Monitoring
    enable_metrics: bool = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    metrics_port: int = int(os.getenv('METRICS_PORT', '9090'))
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_format: str = os.getenv('LOG_FORMAT', 'json')
    log_file_path: str = os.getenv('LOG_FILE_PATH', 'logs/energy_forecasting.log')
    
    # Security
    enable_auth: bool = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
    jwt_secret_key: str = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    jwt_algorithm: str = os.getenv('JWT_ALGORITHM', 'HS256')
    jwt_expiration_hours: int = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
    
    # Performance
    max_concurrent_requests: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '100'))
    request_timeout: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # Caching
    enable_redis_cache: bool = os.getenv('ENABLE_REDIS_CACHE', 'false').lower() == 'true'
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    redis_db: int = int(os.getenv('REDIS_DB', '0'))
    cache_ttl_seconds: int = int(os.getenv('CACHE_TTL_SECONDS', '3600'))

@dataclass
class ClusteringConfig:
    """Clustering analysis configuration"""
    # K-means parameters
    n_clusters: int = int(os.getenv('N_CLUSTERS', '4'))
    kmeans_init: str = os.getenv('KMEANS_INIT', 'k-means++')
    kmeans_n_init: int = int(os.getenv('KMEANS_N_INIT', '10'))
    kmeans_max_iter: int = int(os.getenv('KMEANS_MAX_ITER', '300'))
    
    # DBSCAN parameters
    dbscan_eps: float = float(os.getenv('DBSCAN_EPS', '0.5'))
    dbscan_min_samples: int = int(os.getenv('DBSCAN_MIN_SAMPLES', '5'))
    
    # Feature selection for clustering
    clustering_features: List[str] = field(default_factory=lambda: [
        'avg_consumption', 'std_consumption', 'max_consumption',
        'min_consumption', 'peak_hours', 'is_weekend'
    ])
    
    # Cluster interpretation
    cluster_labels: Dict[str, str] = field(default_factory=lambda: {
        '0': 'Efficient/Standard',
        '1': 'High Peak Consumption',
        '2': 'High Baseline Consumption',
        '3': 'Weekend/Low Activity'
    })

@dataclass
class NLPConfig:
    """Natural Language Processing configuration"""
    # Model selection
    sentiment_model: str = os.getenv('SENTIMENT_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
    summarization_model: str = os.getenv('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn')
    spacy_model: str = os.getenv('SPACY_MODEL', 'en_core_web_sm')
    
    # Processing parameters
    max_text_length: int = int(os.getenv('MAX_TEXT_LENGTH', '512'))
    confidence_threshold: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
    
    # Language support
    supported_languages: List[str] = field(default_factory=lambda: ['en', 'es', 'fr', 'de'])
    default_language: str = os.getenv('DEFAULT_LANGUAGE', 'en')

@dataclass
class ExperimentConfig:
    """Experiment tracking and MLOps configuration"""
    # Experiment tracking
    experiment_tracking_backend: str = os.getenv('EXPERIMENT_TRACKING', 'mlflow')
    mlflow_tracking_uri: str = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name: str = os.getenv('EXPERIMENT_NAME', 'energy_forecasting')
    
    # Model versioning
    model_registry_backend: str = os.getenv('MODEL_REGISTRY', 'mlflow')
    model_version_stage: str = os.getenv('MODEL_VERSION_STAGE', 'staging')
    
    # A/B testing
    enable_ab_testing: bool = os.getenv('ENABLE_AB_TESTING', 'false').lower() == 'true'
    ab_test_traffic_split: float = float(os.getenv('AB_TEST_TRAFFIC_SPLIT', '0.5'))
    
    # Model monitoring
    enable_drift_detection: bool = os.getenv('ENABLE_DRIFT_DETECTION', 'true').lower() == 'true'
    drift_detection_threshold: float = float(os.getenv('DRIFT_DETECTION_THRESHOLD', '0.1'))
    
    # Performance monitoring
    performance_degradation_threshold: float = float(os.getenv('PERFORMANCE_DEGRADATION_THRESHOLD', '0.05'))
    model_retraining_schedule: str = os.getenv('MODEL_RETRAINING_SCHEDULE', 'weekly')

class ConfigManager:
    """Central configuration manager for the Energy Forecasting Platform"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to additional configuration file
        """
        self.config_file = config_file
        self._load_configs()
        self._validate_configs()
        self._setup_logging()
    
    def _load_configs(self):
        """Load all configuration sections"""
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.business = BusinessConfig()
        self.deployment = DeploymentConfig()
        self.clustering = ClusteringConfig()
        self.nlp = NLPConfig()
        self.experiment = ExperimentConfig()
        
        # Load additional config file if provided
        if self.config_file and Path(self.config_file).exists():
            self._load_config_file()
    
    def _load_config_file(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations with file data
            for section, values in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
        except Exception as e:
            logging.warning(f"Could not load config file {self.config_file}: {e}")
    
    def _validate_configs(self):
        """Validate configuration values"""
        validations = []
        
        # Validate paths exist
        for path_attr in ['model_save_path', 'data_path', 'cache_path']:
            path_value = getattr(self.model, path_attr, None) or getattr(self.data, path_attr, None)
            if path_value:
                Path(path_value).mkdir(parents=True, exist_ok=True)
        
        # Validate numeric ranges
        if not 0 < self.model.train_test_split < 1:
            validations.append("train_test_split must be between 0 and 1")
        
        if not 0 < self.model.dropout_rate < 1:
            validations.append("dropout_rate must be between 0 and 1")
        
        if self.model.sequence_length <= 0:
            validations.append("sequence_length must be positive")
        
        if self.model.forecast_horizon <= 0:
            validations.append("forecast_horizon must be positive")
        
        # Validate business logic
        if self.business.peak_start_hour >= self.business.peak_end_hour:
            validations.append("peak_start_hour must be less than peak_end_hour")
        
        # Log validation results
        if validations:
            for validation in validations:
                logging.warning(f"Configuration validation: {validation}")
        else:
            logging.info("All configurations validated successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.deployment.log_level.upper())
        
        # Create logs directory
        log_dir = Path(self.deployment.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.deployment.log_file_path),
                logging.StreamHandler()
            ]
        )
    
    def get_config_summary(self) -> Dict:
        """Get a summary of current configuration"""
        return {
            'environment': self.deployment.environment,
            'debug_mode': self.deployment.debug_mode,
            'model_types': self.model.ensemble_models,
            'sequence_length': self.model.sequence_length,
            'forecast_horizon': self.model.forecast_horizon,
            'enable_clustering': True,
            'enable_nlp': True,
            'api_enabled': self.api.openai_api_key != '',
            'database_configured': self.database.host != 'localhost'
        }
    
    def export_config(self, output_path: str):
        """Export current configuration to JSON file"""
        config_dict = {}
        
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not attr_name.startswith('_') and hasattr(attr, '__dict__'):
                config_dict[attr_name] = attr.__dict__
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logging.info(f"Configuration exported to {output_path}")
    
    def update_config(self, section: str, **kwargs):
        """Update configuration values dynamically"""
        if hasattr(self, section):
            config_obj = getattr(self, section)
            for key, value in kwargs.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    logging.info(f"Updated {section}.{key} = {value}")
                else:
                    logging.warning(f"Unknown configuration key: {section}.{key}")
        else:
            logging.warning(f"Unknown configuration section: {section}")

# Global configuration instance
config = ConfigManager()

# Environment-specific configurations
def get_config_for_environment(env: str) -> ConfigManager:
    """Get configuration for specific environment"""
    config_file = f"config/{env}.json" if env != 'development' else None
    return ConfigManager(config_file)

# Configuration validation decorator
def validate_config(func):
    """Decorator to validate configuration before function execution"""
    def wrapper(*args, **kwargs):
        if not config:
            raise RuntimeError("Configuration not initialized")
        return func(*args, **kwargs)
    return wrapper

# Configuration constants for easy access
MODEL_SAVE_PATH = config.model.model_save_path
DATA_PATH = config.data.data_path
CACHE_PATH = config.data.cache_path
LOG_LEVEL = config.deployment.log_level
SEQUENCE_LENGTH = config.model.sequence_length
FORECAST_HORIZON = config.model.forecast_horizon
ENERGY_COST_KWH = config.business.energy_cost_per_kwh

if __name__ == "__main__":
    # Configuration testing and validation
    print("Energy Forecasting Platform - Configuration Summary")
    print("=" * 50)
    
    summary = config.get_config_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export sample configuration
    config.export_config("config/sample_config.json")
    print("\nSample configuration exported to config/sample_config.json")
