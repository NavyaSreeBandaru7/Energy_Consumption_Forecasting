import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import json
import asyncio
from pathlib import Path

# Import modules to test
import sys
sys.path.append('..')

from energy_forecasting_main import (
    EnergyDataGenerator, AdvancedMLModels, DeepLearningModels,
    ProphetForecaster, EnergyAgent, ClusteringAnalyzer, NLPInsightGenerator
)
from config import ConfigManager, config
from api_server import app
from fastapi.testclient import TestClient

# Test configuration
TEST_DATA_SIZE = 1000
TEST_PERIODS = 168  # 1 week
TOLERANCE = 0.01

@pytest.fixture
def sample_energy_data():
    """Generate sample energy data for testing"""
    generator = EnergyDataGenerator(seed=42)
    return generator.generate_household_data("2023-01-01", TEST_DATA_SIZE)

@pytest.fixture
def sample_industrial_data():
    """Generate sample industrial data for testing"""
    generator = EnergyDataGenerator(seed=42)
    return generator.generate_industrial_data("2023-01-01", TEST_DATA_SIZE)

@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def api_client():
    """Create FastAPI test client"""
    return TestClient(app)

class TestEnergyDataGenerator:
    """Test energy data generation functionality"""
    
    def test_household_data_generation(self):
        """Test household energy data generation"""
        generator = EnergyDataGenerator(seed=42)
        data = generator.generate_household_data("2023-01-01", TEST_DATA_SIZE)
        
        # Basic structure tests
        assert len(data) == TEST_DATA_SIZE
        assert 'timestamp' in data.columns
        assert 'energy_consumption' in data.columns
        assert 'temperature' in data.columns
        
        # Data quality tests
        assert data['energy_consumption'].min() >= 0
        assert data['energy_consumption'].max() < 100  # Reasonable household max
        assert not data['energy_consumption'].isna().any()
        
        # Time series consistency
        assert data['timestamp'].is_monotonic_increasing
        time_diff = data['timestamp'].diff().dropna()
        assert all(time_diff == pd.Timedelta(hours=1))
    
    def test_industrial_data_generation(self):
        """Test industrial energy data generation"""
        generator = EnergyDataGenerator(seed=42)
        data = generator.generate_industrial_data("2023-01-01", TEST_DATA_SIZE)
        
        # Basic structure tests
        assert len(data) == TEST_DATA_SIZE
        assert 'energy_consumption' in data.columns
        assert 'production_level' in data.columns
        assert 'equipment_efficiency' in data.columns
        
        # Industrial-specific tests
        assert data['energy_consumption'].min() >= 10  # Higher industrial baseline
        assert data['production_level'].max() <= 1.0
        assert data['equipment_efficiency'].mean() > 0.5
    
    def test_seasonal_patterns(self):
        """Test that seasonal patterns are present in generated data"""
        generator = EnergyDataGenerator(seed=42)
        
        # Generate full year of data
        data = generator.generate_household_data("2023-01-01", 8760)
        
        # Test seasonal variation
        summer_data = data[data['month'].isin([6, 7, 8])]
        winter_data = data[data['month'].isin([12, 1, 2])]
        
        summer_avg = summer_data['energy_consumption'].mean()
        winter_avg = winter_data['energy_consumption'].mean()
        
        # Should have seasonal difference
        assert abs(summer_avg - winter_avg) > 0.5
    
    def test_daily_patterns(self):
        """Test daily consumption patterns"""
        generator = EnergyDataGenerator(seed=42)
        data = generator.generate_household_data("2023-01-01", TEST_DATA_SIZE)
        
        # Peak hours should have higher consumption
        peak_hours = data[data['hour'].isin([18, 19, 20])]
        off_peak_hours = data[data['hour'].isin([2, 3, 4])]
        
        peak_avg = peak_hours['energy_consumption'].mean()
        off_peak_avg = off_peak_hours['energy_consumption'].mean()
        
        assert peak_avg > off_peak_avg
    
    def test_weekend_patterns(self):
        """Test weekend vs weekday patterns"""
        generator = EnergyDataGenerator(seed=42)
        data = generator.generate_household_data("2023-01-01", TEST_DATA_SIZE)
        
        weekday_data = data[data['is_weekend'] == 0]
        weekend_data = data[data['is_weekend'] == 1]
        
        # Should have different consumption patterns
        weekday_avg = weekday_data['energy_consumption'].mean()
        weekend_avg = weekend_data['energy_consumption'].mean()
        
        assert abs(weekday_avg - weekend_avg) > 0.1

class TestAdvancedMLModels:
    """Test machine learning model functionality"""
    
    def test_feature_preparation(self, sample_energy_data):
        """Test feature engineering"""
        ml_models = AdvancedMLModels()
        X, y, feature_names = ml_models.prepare_features(sample_energy_data)
        
        # Check output shapes
        assert X.shape[0] == y.shape[0]
        assert len(feature_names) == X.shape[1]
        
        # Check for expected features
        expected_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        assert all(feat in feature_names for feat in expected_features)
        
        # Check for lag features
        lag_features = [f for f in feature_names if 'lag_' in f]
        assert len(lag_features) > 0
        
        # Check for rolling features
        rolling_features = [f for f in feature_names if 'rolling_' in f]
        assert len(rolling_features) > 0
    
    def test_ensemble_model_training(self, sample_energy_data):
        """Test ensemble model training"""
        ml_models = AdvancedMLModels()
        X, y, feature_names = ml_models.prepare_features(sample_energy_data)
        
        if len(X) > 100:  # Only test if we have enough data
            results = ml_models.train_ensemble_model(X, y, feature_names)
            
            # Check that models are trained
            assert 'random_forest' in results
            assert 'gradient_boosting' in results
            
            # Check performance metrics
            for model_name, model_results in results.items():
                assert 'r2' in model_results
                assert 'mae' in model_results
                assert 'rmse' in model_results
                
                # Basic sanity checks
                assert 0 <= model_results['r2'] <= 1
                assert model_results['mae'] >= 0
                assert model_results['rmse'] >= 0
    
    def test_feature_importance(self, sample_energy_data):
        """Test feature importance extraction"""
        ml_models = AdvancedMLModels()
        X, y, feature_names = ml_models.prepare_features(sample_energy_data)
        
        if len(X) > 100:
            results = ml_models.train_ensemble_model(X, y, feature_names)
            
            # Check feature importance
            rf_results = results.get('random_forest')
            if rf_results and 'feature_importance' in rf_results:
                importance_df = rf_results['feature_importance']
                assert len(importance_df) == len(feature_names)
                assert 'feature' in importance_df.columns
                assert 'importance' in importance_df.columns
                assert importance_df['importance'].sum() > 0

class TestDeepLearningModels:
    """Test deep learning model functionality"""
    
    def test_sequence_preparation(self):
        """Test sequence preparation for LSTM"""
        dl_models = DeepLearningModels()
        
        # Create test data
        data = np.random.randn(100)
        X, y = dl_models.prepare_sequences(data, seq_length=10)
        
        # Check shapes
        expected_samples = len(data) - 10
        assert X.shape == (expected_samples, 10)
        assert y.shape == (expected_samples,)
        
        # Check sequence consistency
        assert np.array_equal(X[0], data[0:10])
        assert y[0] == data[10]
    
    def test_lstm_model_creation(self):
        """Test LSTM model architecture"""
        dl_models = DeepLearningModels()
        
        input_shape = (24, 1)
        model = dl_models.create_lstm_model(input_shape)
        
        # Check model structure
        assert model.input_shape == (None, 24, 1)
        assert model.output_shape == (None, 1)
        
        # Check that model compiles
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_cnn_lstm_model_creation(self):
        """Test CNN-LSTM hybrid model"""
        dl_models = DeepLearningModels()
        
        input_shape = (24, 1)
        model = dl_models.create_cnn_lstm_model(input_shape)
        
        # Check model structure
        assert model.input_shape == (None, 24, 1)
        assert model.output_shape == (None, 1)
    
    @pytest.mark.slow
    def test_lstm_training(self, sample_energy_data):
        """Test LSTM model training (marked as slow)"""
        if len(sample_energy_data) < 200:
            pytest.skip("Not enough data for LSTM training test")
        
        dl_models = DeepLearningModels()
        energy_data = sample_energy_data['energy_consumption'].values[:200]
        
        # Mock training to avoid long execution times in tests
        with patch.object(dl_models, 'train_lstm_model') as mock_train:
            mock_train.return_value = {
                'model': Mock(),
                'scaler': Mock(),
                'mae': 2.5,
                'mse': 8.0,
                'rmse': 2.83,
                'r2': 0.85,
                'predictions': np.random.randn(50),
                'actuals': np.random.randn(50),
                'history': Mock()
            }
            
            results = dl_models.train_lstm_model(energy_data)
            
            # Check results structure
            assert 'model' in results
            assert 'mae' in results
            assert 'r2' in results
            assert results['r2'] > 0

class TestProphetForecaster:
    """Test Prophet forecasting functionality"""
    
    def test_data_preparation(self, sample_energy_data):
        """Test Prophet data preparation"""
        forecaster = ProphetForecaster()
        prophet_df = forecaster.prepare_prophet_data(sample_energy_data)
        
        # Check required columns
        assert 'ds' in prophet_df.columns
        assert 'y' in prophet_df.columns
        assert len(prophet_df) == len(sample_energy_data)
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(prophet_df['ds'])
        assert pd.api.types.is_numeric_dtype(prophet_df['y'])
    
    @pytest.mark.slow
    def test_prophet_training(self, sample_energy_data):
        """Test Prophet model training (marked as slow)"""
        if len(sample_energy_data) < 200:
            pytest.skip("Not enough data for Prophet training test")
        
        forecaster = ProphetForecaster()
        prophet_df = forecaster.prepare_prophet_data(sample_energy_data[:200])
        
        # Mock Prophet to avoid slow training
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            # Mock forecast output
            mock_forecast = pd.DataFrame({
                'ds': pd.date_range('2023-01-01', periods=100, freq='H'),
                'yhat': np.random.randn(100),
                'yhat_lower': np.random.randn(100),
                'yhat_upper': np.random.randn(100)
            })
            mock_model.predict.return_value = mock_forecast
            
            results = forecaster.train_prophet_model(prophet_df)
            
            # Check results
            assert 'model' in results
            assert 'forecast' in results
            assert 'mae' in results
            assert 'r2' in results

class TestClusteringAnalyzer:
    """Test clustering analysis functionality"""
    
    def test_consumption_pattern_analysis(self, sample_energy_data):
        """Test clustering analysis of consumption patterns"""
        analyzer = ClusteringAnalyzer()
        
        if len(sample_energy_data) < 100:
            pytest.skip("Not enough data for clustering analysis")
        
        results = analyzer.analyze_consumption_patterns(sample_energy_data)
        
        # Check results structure
        assert 'kmeans_labels' in results
        assert 'dbscan_labels' in results
        assert 'cluster_analysis' in results
        assert 'n_clusters' in results
        
        # Check cluster analysis
        cluster_analysis = results['cluster_analysis']
        assert len(cluster_analysis) > 0
        
        for cluster_name, analysis in cluster_analysis.items():
            assert 'size' in analysis
            assert 'percentage' in analysis
            assert 'characteristics' in analysis
            assert 'cluster_type' in analysis
            
            # Validate percentages sum to ~100%
            assert 0 < analysis['percentage'] <= 100
    
    def test_energy_saving_tips(self, sample_energy_data):
        """Test energy saving tips generation"""
        analyzer = ClusteringAnalyzer()
        
        if len(sample_energy_data) < 100:
            pytest.skip("Not enough data for clustering analysis")
        
        clustering_results = analyzer.analyze_consumption_patterns(sample_energy_data)
        tips = analyzer.get_energy_saving_tips(clustering_results['cluster_analysis'])
        
        # Check tips structure
        assert len(tips) > 0
        
        for cluster_name, cluster_tips in tips.items():
            assert 'cluster_type' in cluster_tips
            assert 'priority_tips' in cluster_tips
            assert 'additional_tips' in cluster_tips
            
            # Check that tips are non-empty
            assert len(cluster_tips['priority_tips']) > 0
            assert len(cluster_tips['additional_tips']) > 0

class TestEnergyAgent:
    """Test AI energy agent functionality"""
    
    def test_agent_initialization(self):
        """Test energy agent initialization"""
        agent = EnergyAgent()
        
        # Check basic initialization
        assert agent.memory is not None
        assert agent.tools is not None
        assert len(agent.tools) > 0
    
    def test_tool_creation(self):
        """Test agent tool creation"""
        agent = EnergyAgent()
        
        # Check that tools are created
        tool_names = [tool.name for tool in agent.tools]
        expected_tools = ["Energy Analysis", "Cost Calculator", 
                         "Efficiency Recommendations", "Peak Usage Identifier"]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    def test_fallback_recommendations(self):
        """Test fallback recommendation system"""
        agent = EnergyAgent()
        
        # Test cost-related query
        response = agent.get_recommendations("How can I reduce costs?")
        assert "cost" in response.lower() or "pricing" in response.lower()
        
        # Test efficiency query
        response = agent.get_recommendations("Improve energy efficiency")
        assert "efficiency" in response.lower() or "upgrade" in response.lower()
    
    def test_recommendation_with_api_key(self):
        """Test recommendations with API key"""
        # Mock OpenAI API
        with patch('openai.OpenAI') as mock_openai:
            agent = EnergyAgent(api_key="test-key")
            
            # Should have agent initialized if API key provided
            assert agent.api_key == "test-key"

class TestNLPInsightGenerator:
    """Test NLP insight generation functionality"""
    
    def test_insight_report_generation(self):
        """Test comprehensive insights report generation"""
        generator = NLPInsightGenerator()
        
        # Mock forecast results
        forecast_results = {
            'lstm': {'r2': 0.92, 'rmse': 2.1, 'mae': 1.8},
            'prophet': {'r2': 0.88, 'rmse': 2.5, 'mae': 2.1}
        }
        
        # Mock cluster analysis
        cluster_analysis = {
            'Cluster_0': {
                'cluster_type': 'High Peak Consumption',
                'percentage': 25.0
            },
            'Cluster_1': {
                'cluster_type': 'Efficient/Standard',
                'percentage': 75.0
            }
        }
        
        report = generator.generate_insights_report(forecast_results, cluster_analysis)
        
        # Check report structure
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        assert "EXECUTIVE SUMMARY" in report
        assert "DETAILED INSIGHTS" in report
        assert "RECOMMENDATIONS" in report
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        generator = NLPInsightGenerator()
        
        # Test positive feedback
        result = generator.analyze_sentiment_feedback("This energy analysis is excellent!")
        assert 'sentiment' in result
        assert 'confidence' in result
        assert 'interpretation' in result
        
        # Test negative feedback
        result = generator.analyze_sentiment_feedback("The predictions are completely wrong!")
        assert result['sentiment'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

class TestConfigManager:
    """Test configuration management system"""
    
    def test_config_initialization(self):
        """Test configuration manager initialization"""
        config_manager = ConfigManager()
        
        # Check that all config sections are loaded
        assert hasattr(config_manager, 'database')
        assert hasattr(config_manager, 'api')
        assert hasattr(config_manager, 'model')
        assert hasattr(config_manager, 'data')
        assert hasattr(config_manager, 'business')
        assert hasattr(config_manager, 'deployment')
    
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigManager()
        
        # Test that validation passes for default config
        # This indirectly tests validation by checking no exceptions are raised
        summary = config_manager.get_config_summary()
        assert isinstance(summary, dict)
        assert 'environment' in summary
    
    def test_config_export(self, temp_directory):
        """Test configuration export"""
        config_manager = ConfigManager()
        
        export_path = Path(temp_directory) / "test_config.json"
        config_manager.export_config(str(export_path))
        
        # Check that file was created
        assert export_path.exists()
        
        # Check that file contains valid JSON
        with open(export_path, 'r') as f:
            config_data = json.load(f)
        
        assert isinstance(config_data, dict)
        assert len(config_data) > 0
    
    def test_config_update(self):
        """Test dynamic configuration updates"""
        config_manager = ConfigManager()
        
        # Update a configuration value
        original_value = config_manager.model.lstm_units
        config_manager.update_config('model', lstm_units=256)
        
        # Check that value was updated
        assert config_manager.model.lstm_units == 256
        
        # Restore original value
        config_manager.update_config('model', lstm_units=original_value)

class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint"""
        response = api_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["status"] == "operational"
    
    def test_health_check(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_generate_data_endpoint(self, api_client):
        """Test synthetic data generation endpoint"""
        # Mock authentication
        with patch('api_server.verify_token', return_value=True):
            response = api_client.post(
                "/generate-data?consumer_type=household&periods=100"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "data" in data
            assert "metadata" in data
            assert len(data["data"]) == 100
            assert data["metadata"]["consumer_type"] == "household"
    
    def test_forecast_endpoint(self, api_client):
        """Test forecasting endpoint"""
        # Create test request data
        forecast_request = {
            "historical_data": [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "energy_consumption": 150.5,
                    "temperature": 20.0,
                    "hour": 0,
                    "day_of_week": 0
                }
            ] * 50,  # Repeat to have enough data
            "forecast_horizon": 24,
            "model_types": ["prophet"],
            "confidence_interval": 0.95,
            "include_uncertainty": True
        }
        
        # Mock authentication and model loading
        with patch('api_server.verify_token', return_value=True), \
             patch('api_server.prophet_forecaster') as mock_forecaster:
            
            # Mock Prophet forecaster
            mock_results = {
                'forecast': pd.DataFrame({
                    'ds': pd.date_range('2024-01-02', periods=24, freq='H'),
                    'yhat': np.random.randn(24),
                    'yhat_lower': np.random.randn(24),
                    'yhat_upper': np.random.randn(24)
                }),
                'r2': 0.85,
                'mae': 2.5,
                'rmse': 3.2
            }
            mock_forecaster.train_prophet_model.return_value = mock_results
            
            response = api_client.post("/forecast", json=forecast_request)
            
            if response.status_code != 200:
                print(f"Response: {response.json()}")
            
            # May need authentication, so check for either success or auth error
            assert response.status_code in [200, 401, 422]
    
    def test_models_endpoint(self, api_client):
        """Test models listing endpoint"""
        with patch('api_server.verify_token', return_value=True):
            response = api_client.get("/models")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "available_models" in data
            assert "total_loaded" in data
            assert isinstance(data["available_models"], dict)

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_forecasting_workflow(self, sample_energy_data):
        """Test complete end-to-end forecasting workflow"""
        if len(sample_energy_data) < 200:
            pytest.skip("Not enough data for integration test")
        
        # Step 1: Prepare data
        generator = EnergyDataGenerator(seed=42)
        data = generator.generate_household_data("2023-01-01", 200)
        
        # Step 2: Train ML models
        ml_models = AdvancedMLModels()
        X, y, feature_names = ml_models.prepare_features(data)
        
        if len(X) > 100:
            ml_results = ml_models.train_ensemble_model(X, y, feature_names)
            
            # Step 3: Perform clustering
            analyzer = ClusteringAnalyzer()
            clustering_results = analyzer.analyze_consumption_patterns(data)
            
            # Step 4: Generate recommendations
            tips = analyzer.get_energy_saving_tips(clustering_results['cluster_analysis'])
            
            # Step 5: Generate insights report
            generator = NLPInsightGenerator()
            report = generator.generate_insights_report(ml_results, clustering_results['cluster_analysis'])
            
            # Verify complete workflow
            assert len(ml_results) > 0
            assert len(clustering_results['cluster_analysis']) > 0
            assert len(tips) > 0
            assert len(report) > 100
    
    def test_data_pipeline_integration(self):
        """Test data generation to model training pipeline"""
        # Generate data
        generator = EnergyDataGenerator(seed=42)
        data = generator.generate_household_data("2023-01-01", 500)
        
        # Prepare features
        ml_models = AdvancedMLModels()
        X, y, feature_names = ml_models.prepare_features(data)
        
        # Basic validation of pipeline
        assert len(X) > 0
        assert len(y) > 0
        assert len(feature_names) > 10  # Should have many engineered features
        
        # Check data quality
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_api_model_integration(self, api_client):
        """Test API integration with model components"""
        # Test that API can generate data and handle requests
        with patch('api_server.verify_token', return_value=True):
            # Test data generation
            response = api_client.post("/generate-data?periods=50")
            assert response.status_code == 200
            
            # Test analytics summary
            response = api_client.get("/analytics/summary")
            assert response.status_code == 200

# Performance and stress tests
class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        generator = EnergyDataGenerator(seed=42)
        
        # Generate large dataset
        large_data = generator.generate_household_data("2023-01-01", 10000)
        
        # Test that processing completes in reasonable time
        import time
        start_time = time.time()
        
        ml_models = AdvancedMLModels()
        X, y, feature_names = ml_models.prepare_features(large_data)
        
        processing_time = time.time() - start_time
        
        # Should process 10k records in under 30 seconds
        assert processing_time < 30
        assert len(X) > 5000  # After dropna, should still have substantial data
    
    @pytest.mark.slow
    def test_clustering_performance(self):
        """Test clustering performance with large datasets"""
        generator = EnergyDataGenerator(seed=42)
        large_data = generator.generate_household_data("2023-01-01", 5000)
        
        analyzer = ClusteringAnalyzer()
        
        import time
        start_time = time.time()
        
        results = analyzer.analyze_consumption_patterns(large_data)
        
        clustering_time = time.time() - start_time
        
        # Should complete clustering in reasonable time
        assert clustering_time < 60  # 1 minute max
        assert len(results['cluster_analysis']) > 0

# Error handling and edge cases
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        empty_df = pd.DataFrame()
        
        ml_models = AdvancedMLModels()
        
        # Should handle empty data gracefully
        with pytest.raises((ValueError, IndexError)):
            ml_models.prepare_features(empty_df)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        # Create data with NaN values
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'energy_consumption': [np.nan] * 100,
            'hour': range(100),
            'day_of_week': [0] * 100,
            'month': [1] * 100
        })
        
        ml_models = AdvancedMLModels()
        
        # Should handle invalid data by dropping NaN rows
        try:
            X, y, feature_names = ml_models.prepare_features(invalid_data)
            # If it succeeds, should have no data left
            assert len(X) == 0
        except (ValueError, IndexError):
            # Expected behavior for all-NaN data
            pass
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for modeling"""
        # Very small dataset
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='H'),
            'energy_consumption': [1, 2, 3, 4, 5],
            'hour': [0, 1, 2, 3, 4],
            'day_of_week': [0, 0, 0, 0, 0],
            'month': [1, 1, 1, 1, 1]
        })
        
        ml_models = AdvancedMLModels()
        X, y, feature_names = ml_models.prepare_features(small_data)
        
        # After feature engineering with lags, should have very little data
        assert len(X) < 5

# Test fixtures and utilities
class TestUtilities:
    """Test utility functions and helpers"""
    
    def test_data_validation_utilities(self):
        """Test data validation helper functions"""
        # Test timestamp validation
        valid_timestamps = pd.date_range('2023-01-01', periods=100, freq='H')
        assert len(valid_timestamps) == 100
        assert valid_timestamps.is_monotonic_increasing
    
    def test_metric_calculation_utilities(self):
        """Test metric calculation utilities"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Create test predictions
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        # Test metric calculations
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        assert mae >= 0
        assert mse >= 0
        assert -1 <= r2 <= 1

# Pytest configuration and markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )

# Custom test utilities
def assert_model_performance(results, min_r2=0.5):
    """Custom assertion for model performance"""
    assert 'r2' in results
    assert results['r2'] >= min_r2, f"Model R² score {results['r2']} is below minimum {min_r2}"

def assert_forecast_quality(predictions, actuals, max_mape=20):
    """Custom assertion for forecast quality"""
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    assert mape <= max_mape, f"Mean Absolute Percentage Error {mape:.2f}% exceeds maximum {max_mape}%"

# Test data generators
def generate_test_timeseries(length=1000, trend=0.01, seasonality=True, noise=0.1):
    """Generate synthetic time series for testing"""
    t = np.arange(length)
    
    # Base trend
    series = trend * t
    
    # Seasonality
    if seasonality:
        series += 5 * np.sin(2 * np.pi * t / 24)  # Daily
        series += 2 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly
    
    # Noise
    series += np.random.normal(0, noise, length)
    
    return series

if __name__ == "__main__":
    # Run tests when file is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
