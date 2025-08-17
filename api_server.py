from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import json
from pathlib import Path
import joblib
import pickle
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from contextlib import asynccontextmanager

# Local imports
from config import config
from energy_forecasting_main import (
    EnergyDataGenerator, AdvancedMLModels, DeepLearningModels, 
    ProphetForecaster, EnergyAgent, ClusteringAnalyzer, NLPInsightGenerator
)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration')
PREDICTION_ACCURACY = Gauge('model_prediction_accuracy', 'Model prediction accuracy', ['model_type'])
ACTIVE_MODELS = Gauge('active_models_total', 'Number of active models')

# Security
security = HTTPBearer()

# Logging setup
logging.basicConfig(level=getattr(logging, config.deployment.log_level))
logger = logging.getLogger(__name__)

# Redis cache (optional)
redis_client = None
if config.deployment.enable_redis_cache:
    try:
        redis_client = redis.Redis(
            host=config.deployment.redis_host,
            port=config.deployment.redis_port,
            db=config.deployment.redis_db,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis cache connected successfully")
    except Exception as e:
        logger.warning(f"Redis cache not available: {e}")
        redis_client = None

# Pydantic models for API
class EnergyDataPoint(BaseModel):
    """Single energy consumption data point"""
    timestamp: datetime
    energy_consumption: float = Field(..., gt=0, description="Energy consumption in kWh")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    occupancy: Optional[int] = Field(None, ge=0, le=1, description="Occupancy flag (0 or 1)")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Monday)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day")

class EnergyDataBatch(BaseModel):
    """Batch of energy consumption data"""
    data: List[EnergyDataPoint]
    consumer_type: str = Field("household", regex="^(household|industrial|commercial)$")
    location: Optional[str] = Field(None, description="Geographic location")
    
    @validator('data')
    def validate_data_not_empty(cls, v):
        if not v:
            raise ValueError('Data cannot be empty')
        return v

class ForecastRequest(BaseModel):
    """Forecast request parameters"""
    historical_data: List[EnergyDataPoint]
    forecast_horizon: int = Field(168, gt=0, le=720, description="Forecast horizon in hours")
    model_types: List[str] = Field(["lstm", "prophet"], description="Models to use for forecasting")
    confidence_interval: float = Field(0.95, gt=0.5, lt=1.0, description="Confidence interval")
    include_uncertainty: bool = Field(True, description="Include uncertainty bounds")
    
    @validator('model_types')
    def validate_model_types(cls, v):
        valid_models = ["random_forest", "gradient_boosting", "lstm", "cnn_lstm", "prophet", "arima"]
        for model in v:
            if model not in valid_models:
                raise ValueError(f'Invalid model type: {model}. Valid types: {valid_models}')
        return v

class ForecastResponse(BaseModel):
    """Forecast response with predictions"""
    forecast_id: str
    generated_at: datetime
    forecast_horizon: int
    predictions: List[Dict[str, Union[float, str]]]
    model_performance: Dict[str, Dict[str, float]]
    confidence_intervals: Optional[Dict[str, List[float]]]
    metadata: Dict[str, Union[str, float, int]]

class ClusteringRequest(BaseModel):
    """Clustering analysis request"""
    data: List[EnergyDataPoint]
    clustering_method: str = Field("kmeans", regex="^(kmeans|dbscan|hierarchical)$")
    n_clusters: Optional[int] = Field(4, gt=1, le=10)
    features: Optional[List[str]] = Field(None, description="Features to use for clustering")

class RecommendationRequest(BaseModel):
    """Energy recommendation request"""
    consumption_data: List[EnergyDataPoint]
    consumer_profile: Dict[str, Union[str, float, int]]
    recommendation_type: str = Field("efficiency", regex="^(efficiency|cost|peak|sustainability)$")
    priority_level: str = Field("medium", regex="^(low|medium|high|critical)$")

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    models_loaded: int
    cache_status: str
    database_status: str

# Global model storage
class ModelManager:
    """Manage loaded ML models"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.last_loaded = {}
        
    async def load_model(self, model_type: str, force_reload: bool = False):
        """Load or reload a specific model"""
        if model_type in self.models and not force_reload:
            return self.models[model_type]
        
        try:
            model_path = Path(config.model.model_save_path) / f"{model_type}_model.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models[model_type] = model_data
                self.last_loaded[model_type] = datetime.now()
                self.model_metadata[model_type] = {
                    'loaded_at': self.last_loaded[model_type],
                    'model_path': str(model_path),
                    'size_mb': model_path.stat().st_size / (1024 * 1024)
                }
                
                logger.info(f"Model {model_type} loaded successfully")
                ACTIVE_MODELS.set(len(self.models))
                return model_data
            else:
                logger.warning(f"Model file not found: {model_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            return None
    
    async def get_model(self, model_type: str):
        """Get model, loading if necessary"""
        if model_type not in self.models:
            await self.load_model(model_type)
        return self.models.get(model_type)
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'loaded_models': list(self.models.keys()),
            'model_count': len(self.models),
            'metadata': self.model_metadata
        }

model_manager = ModelManager()

# Initialize components
energy_generator = EnergyDataGenerator()
ml_models = AdvancedMLModels()
dl_models = DeepLearningModels()
prophet_forecaster = ProphetForecaster()
energy_agent = EnergyAgent(config.api.openai_api_key)
clustering_analyzer = ClusteringAnalyzer()
nlp_generator = NLPInsightGenerator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Energy Forecasting API Server")
    
    # Start Prometheus metrics server
    if config.deployment.enable_metrics:
        start_http_server(config.deployment.metrics_port)
        logger.info(f"Metrics server started on port {config.deployment.metrics_port}")
    
    # Preload common models
    await model_manager.load_model("lstm")
    await model_manager.load_model("prophet")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Energy Forecasting API Server")

# FastAPI app initialization
app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Advanced ML-powered energy forecasting platform with real-time predictions",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token (simplified - implement proper JWT validation in production)"""
    if not config.deployment.enable_auth:
        return True
    
    token = credentials.credentials
    # Implement proper JWT verification here
    if token == "demo-token":
        return True
    
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Cache decorator
def cache_result(ttl: int = 3600):
    """Cache API results in Redis"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not redis_client:
                return await func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
            
            result = await func(*args, **kwargs)
            
            try:
                redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
            
            return result
        return wrapper
    return decorator

# Metrics middleware
@app.middleware("http")
async def add_metrics(request, call_next):
    """Add prometheus metrics to all requests"""
    start_time = time.time()
    
    # Count request
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    
    # Process request
    response = await call_next(request)
    
    # Record duration
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    
    return response

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "service": "Energy Forecasting API",
        "version": "2.1.0",
        "status": "operational",
        "documentation": "/docs",
        "metrics": "/metrics" if config.deployment.enable_metrics else None
    }

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    
    # Check models
    model_info = model_manager.get_model_info()
    
    # Check cache
    cache_status = "disabled"
    if redis_client:
        try:
            redis_client.ping()
            cache_status = "healthy"
        except:
            cache_status = "unhealthy"
    
    # Check database (simplified)
    database_status = "not_configured"
    if config.database.host != "localhost":
        database_status = "configured"
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="2.1.0",
        environment=config.deployment.environment,
        models_loaded=model_info['model_count'],
        cache_status=cache_status,
        database_status=database_status
    )

@app.post("/generate-data", tags=["Data"])
async def generate_synthetic_data(
    consumer_type: str = "household",
    periods: int = 8760,
    start_date: str = None,
    authenticated: bool = Depends(verify_token)
):
    """Generate synthetic energy consumption data"""
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        if consumer_type == "household":
            df = energy_generator.generate_household_data(start_date, periods)
        elif consumer_type == "industrial":
            df = energy_generator.generate_industrial_data(start_date, periods)
        else:
            raise HTTPException(status_code=400, detail="Invalid consumer type")
        
        # Convert to JSON serializable format
        data = []
        for _, row in df.iterrows():
            data.append({
                "timestamp": row['timestamp'].isoformat(),
                "energy_consumption": float(row['energy_consumption']),
                "temperature": float(row.get('temperature', 0)),
                "hour": int(row['hour']),
                "day_of_week": int(row['day_of_week']),
                "month": int(row['month'])
            })
        
        return {
            "data": data,
            "metadata": {
                "consumer_type": consumer_type,
                "periods": len(data),
                "start_date": start_date,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
@cache_result(ttl=1800)  # Cache for 30 minutes
async def create_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_token)
):
    """Generate energy consumption forecast using multiple ML models"""
    
    try:
        # Convert request data to DataFrame
        data_records = []
        for point in request.historical_data:
            data_records.append({
                'timestamp': point.timestamp,
                'energy_consumption': point.energy_consumption,
                'temperature': point.temperature or 20.0,
                'hour': point.hour or point.timestamp.hour,
                'day_of_week': point.day_of_week or point.timestamp.weekday(),
                'month': point.timestamp.month,
                'is_weekend': 1 if point.timestamp.weekday() >= 5 else 0
            })
        
        df = pd.DataFrame(data_records)
        
        # Initialize results
        forecast_id = f"forecast_{int(time.time())}"
        predictions = []
        model_performance = {}
        confidence_intervals = {}
        
        # Run forecasting models
        for model_type in request.model_types:
            try:
                if model_type == "prophet":
                    prophet_df = prophet_forecaster.prepare_prophet_data(df)
                    prophet_results = prophet_forecaster.train_prophet_model(prophet_df)
                    
                    # Generate future predictions
                    future_df = prophet_forecaster.model.make_future_dataframe(
                        periods=request.forecast_horizon, freq='H'
                    )
                    forecast = prophet_forecaster.model.predict(future_df)
                    
                    # Extract predictions for forecast horizon
                    future_predictions = forecast.tail(request.forecast_horizon)
                    
                    for _, row in future_predictions.iterrows():
                        predictions.append({
                            'timestamp': row['ds'].isoformat(),
                            'predicted_consumption': float(row['yhat']),
                            'model': 'prophet',
                            'lower_bound': float(row['yhat_lower']) if request.include_uncertainty else None,
                            'upper_bound': float(row['yhat_upper']) if request.include_uncertainty else None
                        })
                    
                    model_performance['prophet'] = {
                        'r2_score': float(prophet_results['r2']),
                        'mae': float(prophet_results['mae']),
                        'rmse': float(prophet_results['rmse'])
                    }
                    
                    if request.include_uncertainty:
                        confidence_intervals['prophet'] = {
                            'lower': future_predictions['yhat_lower'].tolist(),
                            'upper': future_predictions['yhat_upper'].tolist()
                        }
                
                elif model_type == "lstm":
                    # Load or train LSTM model
                    lstm_model = await model_manager.get_model("lstm")
                    if not lstm_model:
                        # Train new LSTM model
                        energy_data = df['energy_consumption'].values
                        lstm_results = dl_models.train_lstm_model(energy_data)
                        lstm_model = lstm_results
                    
                    # Generate LSTM predictions (simplified)
                    last_sequence = df['energy_consumption'].tail(config.model.sequence_length).values
                    lstm_predictions = []
                    
                    # Simple prediction loop (in production, use proper batching)
                    current_sequence = last_sequence.copy()
                    for _ in range(request.forecast_horizon):
                        # Normalize
                        normalized_seq = (current_sequence - current_sequence.mean()) / current_sequence.std()
                        
                        # Predict next value (simplified)
                        next_value = current_sequence[-1] * (1 + np.random.normal(0, 0.05))
                        lstm_predictions.append(max(next_value, 0))
                        
                        # Update sequence
                        current_sequence = np.append(current_sequence[1:], next_value)
                    
                    # Add to predictions
                    start_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
                    for i, pred in enumerate(lstm_predictions):
                        pred_time = start_time + timedelta(hours=i)
                        predictions.append({
                            'timestamp': pred_time.isoformat(),
                            'predicted_consumption': float(pred),
                            'model': 'lstm',
                            'lower_bound': float(pred * 0.9) if request.include_uncertainty else None,
                            'upper_bound': float(pred * 1.1) if request.include_uncertainty else None
                        })
                    
                    model_performance['lstm'] = {
                        'r2_score': 0.85,  # Placeholder - use actual metrics
                        'mae': 2.5,
                        'rmse': 3.2
                    }
                
                # Update model accuracy metrics
                if model_type in model_performance:
                    PREDICTION_ACCURACY.labels(model_type=model_type).set(
                        model_performance[model_type]['r2_score']
                    )
                    
            except Exception as e:
                logger.error(f"Error in {model_type} forecasting: {e}")
                continue
        
        # Schedule background model retraining if needed
        background_tasks.add_task(check_model_performance, model_performance)
        
        response = ForecastResponse(
            forecast_id=forecast_id,
            generated_at=datetime.now(),
            forecast_horizon=request.forecast_horizon,
            predictions=predictions,
            model_performance=model_performance,
            confidence_intervals=confidence_intervals if request.include_uncertainty else None,
            metadata={
                'data_points_used': len(df),
                'models_used': request.model_types,
                'confidence_interval': request.confidence_interval,
                'generated_by': 'Energy Forecasting API v2.1.0'
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")

@app.post("/clustering", tags=["Analytics"])
async def analyze_consumption_patterns(
    request: ClusteringRequest,
    authenticated: bool = Depends(verify_token)
):
    """Analyze energy consumption patterns using clustering"""
    
    try:
        # Convert request data to DataFrame
        data_records = []
        for point in request.data:
            data_records.append({
                'timestamp': point.timestamp,
                'energy_consumption': point.energy_consumption,
                'temperature': point.temperature or 20.0,
                'hour': point.hour or point.timestamp.hour,
                'day_of_week': point.day_of_week or point.timestamp.weekday(),
                'month': point.timestamp.month,
                'is_weekend': 1 if point.timestamp.weekday() >= 5 else 0
            })
        
        df = pd.DataFrame(data_records)
        
        # Perform clustering analysis
        clustering_results = clustering_analyzer.analyze_consumption_patterns(df)
        energy_tips = clustering_analyzer.get_energy_saving_tips(clustering_results['cluster_analysis'])
        
        # Format response
        clusters = []
        for cluster_name, analysis in clustering_results['cluster_analysis'].items():
            clusters.append({
                'cluster_id': cluster_name,
                'cluster_type': analysis['cluster_type'],
                'size': analysis['size'],
                'percentage': analysis['percentage'],
                'characteristics': analysis['characteristics'],
                'recommendations': energy_tips[cluster_name]['priority_tips']
            })
        
        return {
            'clustering_method': request.clustering_method,
            'n_clusters': clustering_results['n_clusters'],
            'clusters': clusters,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_points_analyzed': len(df)
        }
        
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@app.post("/recommendations", tags=["AI Agent"])
async def get_energy_recommendations(
    request: RecommendationRequest,
    authenticated: bool = Depends(verify_token)
):
    """Get AI-powered energy efficiency recommendations"""
    
    try:
        # Convert data for analysis
        data_records = []
        for point in request.consumption_data:
            data_records.append({
                'timestamp': point.timestamp,
                'energy_consumption': point.energy_consumption,
                'temperature': point.temperature or 20.0
            })
        
        df = pd.DataFrame(data_records)
        
        # Generate recommendations using AI agent
        query = f"Provide {request.recommendation_type} recommendations for energy consumption data with {request.priority_level} priority"
        ai_recommendations = energy_agent.get_recommendations(query)
        
        # Analyze consumption patterns
        avg_consumption = df['energy_consumption'].mean()
        peak_consumption = df['energy_consumption'].max()
        consumption_variability = df['energy_consumption'].std()
        
        # Calculate potential savings
        baseline_cost = avg_consumption * config.business.energy_cost_per_kwh * 24 * 30  # Monthly
        potential_savings = baseline_cost * config.business.efficiency_target
        
        # Generate structured recommendations
        recommendations = {
            'ai_analysis': ai_recommendations,
            'consumption_analysis': {
                'average_daily_consumption': float(avg_consumption * 24),
                'peak_consumption': float(peak_consumption),
                'consumption_variability': float(consumption_variability),
                'estimated_monthly_cost': float(baseline_cost),
                'potential_monthly_savings': float(potential_savings)
            },
            'priority_actions': [
                f"Focus on {request.recommendation_type} improvements",
                "Implement smart scheduling for high-energy appliances",
                "Consider time-of-use pricing optimization"
            ],
            'technical_recommendations': [
                "Install smart meters for real-time monitoring",
                "Upgrade to energy-efficient equipment",
                "Implement automated demand response systems"
            ],
            'generated_at': datetime.now().isoformat(),
            'priority_level': request.priority_level
        }
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@app.get("/models", tags=["Models"])
async def list_available_models(authenticated: bool = Depends(verify_token)):
    """List available forecasting models and their status"""
    
    model_info = model_manager.get_model_info()
    
    available_models = {
        'random_forest': 'Ensemble learning for baseline forecasting',
        'gradient_boosting': 'Advanced ensemble with gradient boosting',
        'lstm': 'Deep learning recurrent neural network',
        'cnn_lstm': 'Hybrid CNN-LSTM for pattern recognition',
        'prophet': 'Facebook Prophet for time series forecasting',
        'arima': 'Statistical ARIMA model for time series'
    }
    
    model_status = {}
    for model_name, description in available_models.items():
        is_loaded = model_name in model_info['loaded_models']
        model_status[model_name] = {
            'description': description,
            'status': 'loaded' if is_loaded else 'available',
            'last_loaded': model_info['metadata'].get(model_name, {}).get('loaded_at'),
            'size_mb': model_info['metadata'].get(model_name, {}).get('size_mb')
        }
    
    return {
        'available_models': model_status,
        'total_loaded': model_info['model_count'],
        'last_updated': datetime.now().isoformat()
    }

@app.post("/models/{model_type}/reload", tags=["Models"])
async def reload_model(
    model_type: str,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_token)
):
    """Reload a specific model"""
    
    try:
        # Schedule model reload in background
        background_tasks.add_task(model_manager.load_model, model_type, force_reload=True)
        
        return {
            'message': f'Model {model_type} reload scheduled',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/analytics/summary", tags=["Analytics"])
async def get_analytics_summary(
    days: int = 7,
    authenticated: bool = Depends(verify_token)
):
    """Get summary analytics for the platform"""
    
    # This would typically query a database for real metrics
    # For demo purposes, return mock data
    
    summary = {
        'time_period': f'Last {days} days',
        'total_forecasts_generated': 1247,
        'total_data_points_processed': 89342,
        'average_model_accuracy': 0.87,
        'most_used_model': 'prophet',
        'energy_savings_identified': 23.5,  # Percentage
        'peak_demand_reductions': 15.8,  # Percentage
        'api_requests_count': 5643,
        'error_rate': 0.02,  # 2%
        'average_response_time_ms': 245,
        'top_recommendation_categories': [
            'Equipment Efficiency',
            'Peak Load Management',
            'Renewable Integration',
            'Demand Response'
        ],
        'generated_at': datetime.now().isoformat()
    }
    
    return summary

# Background tasks
async def check_model_performance(model_performance: Dict):
    """Background task to check model performance and trigger retraining if needed"""
    
    for model_name, metrics in model_performance.items():
        if metrics['r2_score'] < config.experiment.performance_degradation_threshold:
            logger.warning(f"Model {model_name} performance degraded: R²={metrics['r2_score']}")
            # Trigger retraining logic here
            
    logger.info("Model performance check completed")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Energy Forecasting API starting up...")
    
    # Create necessary directories
    Path(config.model.model_save_path).mkdir(parents=True, exist_ok=True)
    Path(config.data.data_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Startup completed successfully")

# Main execution
def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "api_server:app",
        host=config.deployment.api_host,
        port=config.deployment.api_port,
        workers=config.deployment.api_workers,
        log_level=config.deployment.log_level.lower(),
        reload=config.deployment.debug_mode
    )

if __name__ == "__main__":
    main()
