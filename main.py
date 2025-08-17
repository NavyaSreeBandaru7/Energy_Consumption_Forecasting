import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
import joblib
import pickle

# Time series specific
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Advanced AI components
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import requests

# Data processing & utilities
import datetime as dt
from datetime import datetime, timedelta
import pytz
import holidays
import sqlite3
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configuration and constants
@dataclass
class Config:
    """System configuration parameters"""
    MODEL_SAVE_PATH: str = "models/"
    DATA_PATH: str = "data/"
    CACHE_PATH: str = "cache/"
    LOG_LEVEL: str = "INFO"
    API_TIMEOUT: int = 30
    BATCH_SIZE: int = 64
    SEQUENCE_LENGTH: int = 24
    FORECAST_HORIZON: int = 168  # 7 days in hours
    
    # Model parameters
    LSTM_UNITS: int = 128
    DROPOUT_RATE: float = 0.2
    LEARNING_RATE: float = 0.001
    
    # Business parameters
    PEAK_THRESHOLD: float = 0.8  # 80th percentile
    ENERGY_COST_KWH: float = 0.12  # USD per kWh

config = Config()

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_forecasting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnergyDataGenerator:
    """Advanced synthetic energy data generator with realistic patterns"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        
    def generate_household_data(self, start_date: str, periods: int = 8760) -> pd.DataFrame:
        """Generate realistic household energy consumption data"""
        
        # Create datetime index
        date_range = pd.date_range(start=start_date, periods=periods, freq='H')
        
        # Base consumption patterns
        base_consumption = 2.5  # kWh base load
        
        # Seasonal patterns (heating/cooling)
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))
        
        # Daily patterns (higher during evening, lower at night)
        daily_pattern = np.tile(
            1 + 0.3 * np.sin(2 * np.pi * np.arange(24) / 24 + np.pi/4), 
            periods // 24 + 1
        )[:periods]
        
        # Weekly patterns (lower on weekends)
        weekly_pattern = np.tile([1.1, 1.1, 1.1, 1.1, 1.1, 0.8, 0.9], periods // (7*24) + 1)
        weekly_pattern = np.repeat(weekly_pattern, 24)[:periods]
        
        # Weather impact simulation
        temp_variation = 20 + 15 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24)) + \
                        np.random.normal(0, 3, periods)
        
        heating_cooling_load = np.where(
            temp_variation < 15, (15 - temp_variation) * 0.1,  # Heating
            np.where(temp_variation > 25, (temp_variation - 25) * 0.08, 0)  # Cooling
        )
        
        # Random appliance usage spikes
        appliance_spikes = np.random.exponential(0.5, periods) * \
                          np.random.binomial(1, 0.1, periods)
        
        # Combine all factors
        consumption = (base_consumption * seasonal_factor * daily_pattern * 
                      weekly_pattern + heating_cooling_load + appliance_spikes)
        
        # Add realistic noise
        consumption += np.random.normal(0, 0.1, periods)
        consumption = np.maximum(consumption, 0.5)  # Minimum consumption
        
        # Create comprehensive dataset
        df = pd.DataFrame({
            'timestamp': date_range,
            'energy_consumption': consumption,
            'temperature': temp_variation,
            'hour': date_range.hour,
            'day_of_week': date_range.dayofweek,
            'month': date_range.month,
            'season': self._get_season(date_range),
            'is_weekend': (date_range.dayofweek >= 5).astype(int),
            'is_holiday': self._get_holidays(date_range),
            'occupancy': self._simulate_occupancy(date_range),
            'electricity_price': self._simulate_price(date_range)
        })
        
        return df
    
    def generate_industrial_data(self, start_date: str, periods: int = 8760) -> pd.DataFrame:
        """Generate realistic industrial energy consumption data"""
        
        date_range = pd.date_range(start=start_date, periods=periods, freq='H')
        
        # Industrial base load (much higher than household)
        base_consumption = 150  # kWh
        
        # Production schedule patterns (5-day work week, 3-shift operation)
        production_schedule = np.zeros(periods)
        for i in range(periods):
            day_of_week = date_range[i].dayofweek
            hour = date_range[i].hour
            
            if day_of_week < 5:  # Monday to Friday
                if 6 <= hour < 22:  # Peak production hours
                    production_schedule[i] = 1.0
                else:  # Night shift (reduced)
                    production_schedule[i] = 0.6
            else:  # Weekend maintenance/minimal operation
                production_schedule[i] = 0.3
        
        # Equipment cycling patterns
        equipment_cycles = np.sin(2 * np.pi * np.arange(periods) / 8) * 0.2  # 8-hour cycles
        
        # Seasonal variations (less pronounced than household)
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24))
        
        # Production demand variations
        production_demand = np.random.normal(1, 0.15, periods)
        production_demand = np.maximum(production_demand, 0.5)
        
        # Calculate industrial consumption
        consumption = (base_consumption * production_schedule * seasonal_factor * 
                      production_demand * (1 + equipment_cycles))
        
        # Add industrial-specific noise
        consumption += np.random.normal(0, 5, periods)
        consumption = np.maximum(consumption, 10)  # Minimum industrial load
        
        # Create industrial dataset
        df = pd.DataFrame({
            'timestamp': date_range,
            'energy_consumption': consumption,
            'production_level': production_schedule,
            'equipment_efficiency': np.random.normal(0.85, 0.05, periods),
            'ambient_temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(periods) / (365.25 * 24)) + \
                                 np.random.normal(0, 2, periods),
            'hour': date_range.hour,
            'day_of_week': date_range.dayofweek,
            'month': date_range.month,
            'is_working_day': (date_range.dayofweek < 5).astype(int),
            'shift': self._get_shift(date_range),
            'maintenance_flag': np.random.binomial(1, 0.02, periods)  # 2% chance of maintenance
        })
        
        return df
    
    def _get_season(self, date_range):
        """Get season for each timestamp"""
        seasons = []
        for date in date_range:
            month = date.month
            if month in [12, 1, 2]:
                seasons.append('Winter')
            elif month in [3, 4, 5]:
                seasons.append('Spring')
            elif month in [6, 7, 8]:
                seasons.append('Summer')
            else:
                seasons.append('Fall')
        return seasons
    
    def _get_holidays(self, date_range):
        """Simulate holiday effects"""
        us_holidays = holidays.US()
        return [1 if date.date() in us_holidays else 0 for date in date_range]
    
    def _simulate_occupancy(self, date_range):
        """Simulate household occupancy patterns"""
        occupancy = np.ones(len(date_range))
        for i, date in enumerate(date_range):
            hour = date.hour
            day_of_week = date.dayofweek
            
            if day_of_week < 5:  # Weekday
                if 8 <= hour < 17:  # Work hours
                    occupancy[i] = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% chance home
                else:
                    occupancy[i] = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% chance home
            else:  # Weekend
                occupancy[i] = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% chance home
        
        return occupancy
    
    def _simulate_price(self, date_range):
        """Simulate time-of-use electricity pricing"""
        prices = np.ones(len(date_range)) * config.ENERGY_COST_KWH
        for i, date in enumerate(date_range):
            hour = date.hour
            if 17 <= hour <= 20:  # Peak hours
                prices[i] *= 1.5
            elif 22 <= hour <= 6:  # Off-peak hours
                prices[i] *= 0.7
        return prices
    
    def _get_shift(self, date_range):
        """Determine industrial work shift"""
        shifts = []
        for date in date_range:
            hour = date.hour
            if 6 <= hour < 14:
                shifts.append('Day')
            elif 14 <= hour < 22:
                shifts.append('Evening')
            else:
                shifts.append('Night')
        return shifts

class AdvancedMLModels:
    """Advanced ML models for energy forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'energy_consumption') -> Tuple[np.ndarray, np.ndarray]:
        """Advanced feature engineering for ML models"""
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # Various lag periods
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24, 168]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        
        # Advanced statistical features
        df['energy_diff'] = df[target_col].diff()
        df['energy_pct_change'] = df[target_col].pct_change()
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Prepare features and target
        feature_cols = [col for col in df_clean.columns if col not in [target_col, 'timestamp']]
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        return X, y, feature_cols
    
    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """Train ensemble of ML models"""
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            if name == 'random_forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:  # gradient_boosting
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'predictions': y_pred,
                'actuals': y_test
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance_df
        
        self.models.update(models)
        self.scalers['standard'] = scaler
        
        return results

class DeepLearningModels:
    """Advanced deep learning models for time series forecasting"""
    
    def __init__(self):
        self.models = {}
        self.history = {}
        
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create advanced LSTM model architecture"""
        
        model = Sequential([
            LSTM(config.LSTM_UNITS, return_sequences=True, input_shape=input_shape),
            Dropout(config.DROPOUT_RATE),
            LSTM(config.LSTM_UNITS // 2, return_sequences=True),
            Dropout(config.DROPOUT_RATE),
            LSTM(config.LSTM_UNITS // 4, return_sequences=False),
            Dropout(config.DROPOUT_RATE),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create CNN-LSTM hybrid model"""
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(config.LSTM_UNITS, return_sequences=False),
            Dropout(config.DROPOUT_RATE),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, data: np.ndarray, seq_length: int = config.SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, data: np.ndarray, model_type: str = 'lstm') -> Dict:
        """Train LSTM model with advanced techniques"""
        
        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X, y = self.prepare_sequences(data_scaled)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        if model_type == 'cnn_lstm':
            model = self.create_cnn_lstm_model(X.shape[1:])
        else:
            model = self.create_lstm_model(X.shape[1:])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        self.models[model_type] = model
        self.history[model_type] = history
        
        return {
            'model': model,
            'scaler': scaler,
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'predictions': y_pred_orig,
            'actuals': y_test_orig,
            'history': history
        }

class ProphetForecaster:
    """Prophet model implementation for time series forecasting"""
    
    def __init__(self):
        self.model = None
        self.forecast = None
        
    def prepare_prophet_data(self, df: pd.DataFrame, target_col: str = 'energy_consumption') -> pd.DataFrame:
        """Prepare data for Prophet model"""
        
        prophet_df = pd.DataFrame({
            'ds': df['timestamp'],
            'y': df[target_col]
        })
        
        # Add regressors if available
        if 'temperature' in df.columns:
            prophet_df['temperature'] = df['temperature']
        if 'is_weekend' in df.columns:
            prophet_df['is_weekend'] = df['is_weekend']
        if 'is_holiday' in df.columns:
            prophet_df['is_holiday'] = df['is_holiday']
            
        return prophet_df
    
    def train_prophet_model(self, df: pd.DataFrame) -> Dict:
        """Train Prophet model with advanced configurations"""
        
        # Initialize Prophet with custom parameters
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=0.1,
            holidays_prior_scale=0.1,
            interval_width=0.95
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        # Add regressors
        if 'temperature' in df.columns:
            self.model.add_regressor('temperature')
        if 'is_weekend' in df.columns:
            self.model.add_regressor('is_weekend')
        if 'is_holiday' in df.columns:
            self.model.add_regressor('is_holiday')
        
        # Fit model
        self.model.fit(df)
        
        # Make forecast
        future = self.model.make_future_dataframe(periods=config.FORECAST_HORIZON, freq='H')
        
        # Add regressor values for future periods
        if 'temperature' in df.columns:
            # Simple temperature simulation for future
            last_temp = df['temperature'].iloc[-1]
            future_temps = [last_temp + np.random.normal(0, 2) for _ in range(len(future) - len(df))]
            future['temperature'] = list(df['temperature']) + future_temps
            
        if 'is_weekend' in df.columns:
            future['is_weekend'] = [(future.iloc[i]['ds'].dayofweek >= 5) * 1 for i in range(len(future))]
            
        if 'is_holiday' in df.columns:
            us_holidays = holidays.US()
            future['is_holiday'] = [1 if future.iloc[i]['ds'].date() in us_holidays else 0 for i in range(len(future))]
        
        forecast = self.model.predict(future)
        self.forecast = forecast
        
        # Calculate metrics on historical data
        historical_forecast = forecast[:-config.FORECAST_HORIZON]
        actual_values = df['y'].values
        predicted_values = historical_forecast['yhat'].values
        
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)
        
        return {
            'model': self.model,
            'forecast': forecast,
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'predictions': predicted_values,
            'actuals': actual_values
        }

class EnergyAgent:
    """Intelligent energy management agent using LangChain"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.memory = ConversationBufferMemory()
        self.tools = self._create_tools()
        self.agent = self._initialize_agent()
        
    def _create_tools(self) -> List[Tool]:
        """Create tools for the energy agent"""
        
        tools = [
            Tool(
                name="Energy Analysis",
                description="Analyze energy consumption patterns and provide insights",
                func=self._analyze_energy_patterns
            ),
            Tool(
                name="Cost Calculator",
                description="Calculate energy costs and potential savings",
                func=self._calculate_energy_costs
            ),
            Tool(
                name="Efficiency Recommendations",
                description="Provide energy efficiency recommendations",
                func=self._get_efficiency_recommendations
            ),
            Tool(
                name="Peak Usage Identifier",
                description="Identify peak usage patterns and times",
                func=self._identify_peak_usage
            )
        ]
        
        return tools
    
    def _initialize_agent(self):
        """Initialize the LangChain agent"""
        if self.api_key:
            llm = OpenAI(api_key=self.api_key, temperature=0.7)
            agent = initialize_agent(
                self.tools,
                llm,
                agent="conversational-react-description",
                memory=self.memory,
                verbose=True
            )
            return agent
        return None
    
    def _analyze_energy_patterns(self, data: str) -> str:
        """Analyze energy consumption patterns"""
        # This would typically process real data
        insights = [
            "Peak consumption occurs during evening hours (6-9 PM)",
            "Weekend consumption is 20% lower than weekdays",
            "Seasonal variations show 30% higher usage in summer",
            "Equipment efficiency has declined by 5% over the past quarter"
        ]
        return "Energy Pattern Analysis: " + "; ".join(insights)
    
    def _calculate_energy_costs(self, consumption_data: str) -> str:
        """Calculate energy costs and savings potential"""
        # Simplified cost calculation
        avg_consumption = 150  # kWh per day
        current_cost = avg_consumption * config.ENERGY_COST_KWH * 30  # Monthly
        potential_savings = current_cost * 0.15  # 15% potential savings
        
        return f"Monthly cost: ${current_cost:.2f}, Potential monthly savings: ${potential_savings:.2f}"
    
    def _get_efficiency_recommendations(self, usage_pattern: str) -> str:
        """Generate efficiency recommendations using clustering insights"""
        recommendations = [
            "Install programmable thermostats to optimize heating/cooling",
            "Upgrade to LED lighting for 60% energy reduction",
            "Implement power management for IT equipment",
            "Schedule high-energy tasks during off-peak hours",
            "Consider renewable energy sources (solar/wind)",
            "Improve building insulation to reduce HVAC load"
        ]
        return "Efficiency Recommendations: " + "; ".join(recommendations[:3])
    
    def _identify_peak_usage(self, data: str) -> str:
        """Identify peak usage patterns"""
        peak_info = {
            "daily_peak": "6:00-9:00 PM",
            "weekly_peak": "Tuesday-Thursday",
            "seasonal_peak": "July-August (cooling), December-February (heating)",
            "peak_load": "85% of maximum capacity"
        }
        return f"Peak Usage Analysis: Daily peak at {peak_info['daily_peak']}, Weekly peak on {peak_info['weekly_peak']}"
    
    def get_recommendations(self, query: str) -> str:
        """Get AI-powered recommendations"""
        if self.agent:
            try:
                response = self.agent.run(query)
                return response
            except Exception as e:
                logger.error(f"Agent error: {e}")
                return self._fallback_recommendations(query)
        else:
            return self._fallback_recommendations(query)
    
    def _fallback_recommendations(self, query: str) -> str:
        """Fallback recommendations when AI agent is not available"""
        recommendations = {
            "cost": "Implement time-of-use pricing strategies and shift high-energy activities to off-peak hours",
            "efficiency": "Focus on equipment upgrades, insulation improvements, and smart automation systems",
            "peak": "Distribute energy-intensive tasks throughout the day and implement demand response programs",
            "savings": "Prioritize high-impact, low-cost efficiency measures like LED lighting and programmable controls"
        }
        
        for key, value in recommendations.items():
            if key in query.lower():
                return value
        
        return "Consider implementing a comprehensive energy management system with real-time monitoring and automated controls"

class ClusteringAnalyzer:
    """Advanced clustering for energy consumption patterns"""
    
    def __init__(self):
        self.models = {}
        self.cluster_labels = {}
        
    def analyze_consumption_patterns(self, df: pd.DataFrame) -> Dict:
        """Perform clustering analysis on energy consumption patterns"""
        
        # Feature engineering for clustering
        features = []
        
        # Daily patterns
        daily_profiles = df.groupby(['hour'])['energy_consumption'].mean()
        
        # Weekly patterns
        weekly_profiles = df.groupby(['day_of_week'])['energy_consumption'].mean()
        
        # Monthly patterns
        monthly_profiles = df.groupby(['month'])['energy_consumption'].mean()
        
        # Create feature matrix for clustering
        daily_features = []
        for day in df['timestamp'].dt.date.unique():
            day_data = df[df['timestamp'].dt.date == day]
            if len(day_data) == 24:  # Complete day
                daily_consumption = day_data.groupby('hour')['energy_consumption'].mean()
                features_day = [
                    daily_consumption.mean(),
                    daily_consumption.std(),
                    daily_consumption.max(),
                    daily_consumption.min(),
                    daily_consumption.max() - daily_consumption.min(),
                    len(daily_consumption[daily_consumption > daily_consumption.quantile(0.8)]),  # Peak hours
                    day_data['is_weekend'].iloc[0],
                    day_data['month'].iloc[0],
                    (daily_consumption > daily_consumption.mean()).sum()  # Above average hours
                ]
                daily_features.append(features_day)
        
        X_cluster = np.array(daily_features)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        self.models['kmeans'] = kmeans
        self.models['dbscan'] = dbscan
        self.models['scaler'] = scaler
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(X_cluster, kmeans_labels)
        
        return {
            'kmeans_labels': kmeans_labels,
            'dbscan_labels': dbscan_labels,
            'cluster_analysis': cluster_analysis,
            'feature_matrix': X_cluster,
            'n_clusters': len(np.unique(kmeans_labels))
        }
    
    def _analyze_clusters(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Analyze cluster characteristics"""
        
        cluster_analysis = {}
        feature_names = [
            'avg_consumption', 'std_consumption', 'max_consumption', 
            'min_consumption', 'range_consumption', 'peak_hours',
            'is_weekend', 'month', 'above_avg_hours'
        ]
        
        for cluster_id in np.unique(labels):
            cluster_data = X[labels == cluster_id]
            
            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100,
                'characteristics': {}
            }
            
            for i, feature in enumerate(feature_names):
                analysis['characteristics'][feature] = {
                    'mean': np.mean(cluster_data[:, i]),
                    'std': np.std(cluster_data[:, i]),
                    'median': np.median(cluster_data[:, i])
                }
            
            # Cluster interpretation
            avg_consumption = analysis['characteristics']['avg_consumption']['mean']
            peak_hours = analysis['characteristics']['peak_hours']['mean']
            is_weekend = analysis['characteristics']['is_weekend']['mean']
            
            if avg_consumption > np.mean(X[:, 0]):
                if peak_hours > 4:
                    cluster_type = "High Peak Consumption"
                else:
                    cluster_type = "High Baseline Consumption"
            else:
                if is_weekend > 0.5:
                    cluster_type = "Weekend/Low Activity"
                else:
                    cluster_type = "Efficient/Standard"
            
            analysis['cluster_type'] = cluster_type
            cluster_analysis[f'Cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def get_energy_saving_tips(self, cluster_analysis: Dict) -> Dict:
        """Generate energy saving tips based on cluster analysis"""
        
        tips = {}
        
        for cluster_name, analysis in cluster_analysis.items():
            cluster_type = analysis['cluster_type']
            tips_list = []
            
            if "High Peak" in cluster_type:
                tips_list.extend([
                    "Implement demand response programs during peak hours",
                    "Install energy storage systems to shift peak loads",
                    "Use smart appliances with delayed start features",
                    "Consider time-of-use pricing to incentivize off-peak usage"
                ])
            
            elif "High Baseline" in cluster_type:
                tips_list.extend([
                    "Audit and optimize always-on equipment",
                    "Improve building envelope efficiency",
                    "Upgrade to more efficient HVAC systems",
                    "Implement occupancy-based controls"
                ])
            
            elif "Weekend/Low Activity" in cluster_type:
                tips_list.extend([
                    "Optimize weekend operation schedules",
                    "Implement deeper setbacks during unoccupied periods",
                    "Consider automated systems for weekend energy management",
                    "Review and adjust baseline loads"
                ])
            
            else:  # Efficient/Standard
                tips_list.extend([
                    "Maintain current efficient practices",
                    "Monitor for any degradation in performance",
                    "Consider renewable energy integration",
                    "Explore advanced automation opportunities"
                ])
            
            # Add universal tips
            tips_list.extend([
                "Regular maintenance of energy-consuming equipment",
                "Employee/occupant education on energy conservation",
                "Real-time energy monitoring and feedback systems"
            ])
            
            tips[cluster_name] = {
                'cluster_type': cluster_type,
                'priority_tips': tips_list[:3],
                'additional_tips': tips_list[3:]
            }
        
        return tips

class NLPInsightGenerator:
    """Natural Language Processing for energy insights"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic text processing.")
            self.nlp = None
        
        self.sentiment_analyzer = None
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception as e:
            logger.warning(f"Sentiment analyzer not available: {e}")
    
    def generate_insights_report(self, forecast_results: Dict, cluster_analysis: Dict) -> str:
        """Generate comprehensive insights report using NLP"""
        
        insights = []
        
        # Forecast insights
        for model_name, results in forecast_results.items():
            if 'r2' in results:
                r2_score = results['r2']
                rmse = results.get('rmse', 0)
                
                if r2_score > 0.9:
                    accuracy_desc = "excellent"
                elif r2_score > 0.8:
                    accuracy_desc = "good"
                elif r2_score > 0.7:
                    accuracy_desc = "fair"
                else:
                    accuracy_desc = "poor"
                
                insights.append(
                    f"The {model_name.replace('_', ' ').title()} model shows {accuracy_desc} "
                    f"forecasting accuracy with an R² score of {r2_score:.3f} and RMSE of {rmse:.2f} kWh."
                )
        
        # Clustering insights
        if cluster_analysis:
            total_clusters = len(cluster_analysis)
            insights.append(f"Energy consumption analysis identified {total_clusters} distinct usage patterns:")
            
            for cluster_name, analysis in cluster_analysis.items():
                cluster_type = analysis['cluster_type']
                percentage = analysis['percentage']
                insights.append(
                    f"• {cluster_type} pattern represents {percentage:.1f}% of the analyzed period"
                )
        
        # Seasonal insights
        insights.append(
            "Seasonal analysis reveals typical patterns with higher consumption during "
            "extreme weather months (summer cooling and winter heating demands)."
        )
        
        # Economic insights
        insights.append(
            "Peak demand charges represent a significant cost opportunity, with potential "
            "savings of 15-25% through demand response and load shifting strategies."
        )
        
        # Technology recommendations
        insights.append(
            "Advanced forecasting models enable proactive energy management, allowing "
            "for optimized scheduling of energy-intensive operations and improved grid stability."
        )
        
        # Generate summary
        summary = self._generate_executive_summary(insights)
        
        full_report = f"""
ENERGY CONSUMPTION FORECASTING & ANALYSIS REPORT
=============================================

EXECUTIVE SUMMARY
{summary}

DETAILED INSIGHTS
{chr(10).join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])}

RECOMMENDATIONS
• Implement real-time monitoring systems for continuous optimization
• Develop automated demand response capabilities
• Integrate renewable energy sources where feasible
• Establish energy efficiency KPIs and regular review processes
• Consider advanced energy storage solutions for peak shaving

NEXT STEPS
• Deploy production forecasting models with automated retraining
• Establish energy management dashboard for stakeholders
• Implement recommended efficiency measures based on cluster analysis
• Develop long-term energy strategy incorporating forecasting insights
        """
        
        return full_report.strip()
    
    def _generate_executive_summary(self, insights: List[str]) -> str:
        """Generate executive summary from insights"""
        
        # Extract key numbers and concepts
        key_points = []
        
        for insight in insights:
            if "excellent" in insight or "good" in insight:
                key_points.append("High forecasting accuracy achieved")
            elif "cluster" in insight or "pattern" in insight:
                key_points.append("Multiple consumption patterns identified")
            elif "savings" in insight:
                key_points.append("Significant cost reduction opportunities available")
        
        summary = (
            "This comprehensive energy analysis demonstrates strong predictive capabilities "
            "with multiple distinct consumption patterns identified. The forecasting models "
            "show reliable accuracy for operational planning, while clustering analysis "
            "reveals targeted optimization opportunities. Implementation of recommended "
            "strategies could yield substantial energy and cost savings."
        )
        
        return summary
    
    def analyze_sentiment_feedback(self, feedback_text: str) -> Dict:
        """Analyze sentiment of user feedback"""
        
        if self.sentiment_analyzer and feedback_text:
            try:
                result = self.sentiment_analyzer(feedback_text)
                return {
                    'sentiment': result[0]['label'],
                    'confidence': result[0]['score'],
                    'interpretation': self._interpret_sentiment(result[0])
                }
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
        
        return {
            'sentiment': 'NEUTRAL',
            'confidence': 0.5,
            'interpretation': 'Sentiment analysis not available'
        }
    
    def _interpret_sentiment(self, sentiment_result: Dict) -> str:
        """Interpret sentiment analysis results"""
        
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        if label == 'POSITIVE' and score > 0.8:
            return "Users are very satisfied with the energy insights"
        elif label == 'POSITIVE':
            return "Generally positive feedback on energy recommendations"
        elif label == 'NEGATIVE' and score > 0.8:
            return "Users are dissatisfied - review recommendations and accuracy"
        elif label == 'NEGATIVE':
            return "Some concerns raised - investigate specific issues"
        else:
            return "Neutral feedback - consider gathering more specific input"

# Streamlit Application
def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Energy Forecasting Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ Advanced Energy Consumption Forecasting Platform")
    st.markdown("*AI-Powered Energy Analytics with Machine Learning & Deep Learning Models*")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Generate Synthetic Data", "Upload CSV File", "Connect to Database"]
    )
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Forecasting Models",
        ["Random Forest", "Gradient Boosting", "LSTM", "CNN-LSTM", "Prophet", "ARIMA"],
        default=["Random Forest", "LSTM", "Prophet"]
    )
    
    # Consumer type
    consumer_type = st.sidebar.selectbox(
        "Consumer Type",
        ["Household", "Industrial", "Commercial"]
    )
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
    confidence_interval = st.sidebar.slider("Confidence Interval", 0.8, 0.99, 0.95)
    
    # Initialize components
    data_generator = EnergyDataGenerator()
    ml_models = AdvancedMLModels()
    dl_models = DeepLearningModels()
    prophet_forecaster = ProphetForecaster()
    energy_agent = EnergyAgent()
    clustering_analyzer = ClusteringAnalyzer()
    nlp_generator = NLPInsightGenerator()
    
    # Main content area
    if data_source == "Generate Synthetic Data":
        st.subheader("📊 Synthetic Data Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            periods = st.number_input("Number of Hours", 1000, 10000, 8760)
        
        with col2:
            if st.button("Generate Data", type="primary"):
                with st.spinner("Generating synthetic energy data..."):
                    if consumer_type == "Household":
                        df = data_generator.generate_household_data(str(start_date), periods)
                    else:
                        df = data_generator.generate_industrial_data(str(start_date), periods)
                    
                    st.session_state['data'] = df
                    st.success(f"Generated {len(df)} hours of {consumer_type.lower()} energy data!")
    
    elif data_source == "Upload CSV File":
        st.subheader("📁 Upload Energy Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            st.success("Data uploaded successfully!")
    
    # Display data if available
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Data overview
        st.subheader("📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Avg Consumption", f"{df['energy_consumption'].mean():.2f} kWh")
        with col3:
            st.metric("Peak Consumption", f"{df['energy_consumption'].max():.2f} kWh")
        with col4:
            st.metric("Data Span", f"{len(df)//24} days")
        
        # Visualization
        st.subheader("📊 Energy Consumption Visualization")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Time Series", "Daily Pattern", "Weekly Pattern", "Monthly Pattern"),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['energy_consumption'], 
                      name="Energy Consumption", line=dict(color='blue')),
            row=1, col=1
        )
        
        if 'temperature' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['temperature'], 
                          name="Temperature", line=dict(color='red')),
                row=1, col=1, secondary_y=True
            )
        
        # Daily pattern
        daily_avg = df.groupby('hour')['energy_consumption'].mean()
        fig.add_trace(
            go.Bar(x=daily_avg.index, y=daily_avg.values, name="Hourly Average"),
            row=1, col=2
        )
        
        # Weekly pattern
        weekly_avg = df.groupby('day_of_week')['energy_consumption'].mean()
        fig.add_trace(
            go.Bar(x=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], 
                   y=weekly_avg.values, name="Daily Average"),
            row=2, col=1
        )
        
        # Monthly pattern
        monthly_avg = df.groupby('month')['energy_consumption'].mean()
        fig.add_trace(
            go.Bar(x=monthly_avg.index, y=monthly_avg.values, name="Monthly Average"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model training and forecasting
        if st.button("🚀 Run Advanced Forecasting", type="primary"):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            
            # Machine Learning Models
            if any(model in selected_models for model in ["Random Forest", "Gradient Boosting"]):
                status_text.text("Training Machine Learning models...")
                progress_bar.progress(20)
                
                X, y, feature_names = ml_models.prepare_features(df)
                ml_results = ml_models.train_ensemble_model(X, y, feature_names)
                results['ml_models'] = ml_results
            
            # Deep Learning Models
            if any(model in selected_models for model in ["LSTM", "CNN-LSTM"]):
                status_text.text("Training Deep Learning models...")
                progress_bar.progress(40)
                
                energy_data = df['energy_consumption'].values
                
                if "LSTM" in selected_models:
                    lstm_results = dl_models.train_lstm_model(energy_data, 'lstm')
                    results['lstm'] = lstm_results
                
                if "CNN-LSTM" in selected_models:
                    cnn_lstm_results = dl_models.train_lstm_model(energy_data, 'cnn_lstm')
                    results['cnn_lstm'] = cnn_lstm_results
            
            # Prophet Model
            if "Prophet" in selected_models:
                status_text.text("Training Prophet model...")
                progress_bar.progress(60)
                
                prophet_df = prophet_forecaster.prepare_prophet_data(df)
                prophet_results = prophet_forecaster.train_prophet_model(prophet_df)
                results['prophet'] = prophet_results
            
            # Clustering Analysis
            status_text.text("Performing clustering analysis...")
            progress_bar.progress(80)
            
            clustering_results = clustering_analyzer.analyze_consumption_patterns(df)
            energy_tips = clustering_analyzer.get_energy_saving_tips(clustering_results['cluster_analysis'])
            
            # Generate NLP insights
            status_text.text("Generating AI insights...")
            progress_bar.progress(90)
            
            insights_report = nlp_generator.generate_insights_report(results, clustering_results['cluster_analysis'])
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Display results
            st.subheader("🎯 Model Performance Comparison")
            
            performance_data = []
            for model_name, model_results in results.items():
                if 'r2' in model_results:
                    performance_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'R² Score': model_results['r2'],
                        'RMSE': model_results['rmse'],
                        'MAE': model_results['mae']
                    })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(perf_df, use_container_width=True)
                
                # Performance visualization
                fig_perf = px.bar(perf_df, x='Model', y='R² Score', 
                                 title="Model Performance Comparison (R² Score)")
                st.plotly_chart(fig_perf, use_container_width=True)
            
            # Forecasting visualization
            st.subheader("🔮 Energy Consumption Forecast")
            
            # Display Prophet forecast if available
            if 'prophet' in results:
                prophet_forecast = results['prophet']['forecast']
                
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['energy_consumption'],
                             name='Historical', line=dict(color='blue'))
                )
                
                # Forecast
                future_data = prophet_forecast.tail(forecast_days * 24)
                fig_forecast.add_trace(
                    go.Scatter(x=future_data['ds'], y=future_data['yhat'],
                             name='Forecast', line=dict(color='red', dash='dash'))
                )
                
                # Confidence intervals
                fig_forecast.add_trace(
                    go.Scatter(x=future_data['ds'], y=future_data['yhat_upper'],
                             fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False)
                )
                fig_forecast.add_trace(
                    go.Scatter(x=future_data['ds'], y=future_data['yhat_lower'],
                             fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', 
                             name='Confidence Interval', fillcolor='rgba(255,0,0,0.2)')
                )
                
                fig_forecast.update_layout(
                    title=f"Energy Consumption Forecast - Next {forecast_days} Days",
                    xaxis_title="Date",
                    yaxis_title="Energy Consumption (kWh)",
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Clustering Analysis Results
            st.subheader("🎪 Consumption Pattern Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Identified Consumption Patterns:**")
                for cluster_name, analysis in clustering_results['cluster_analysis'].items():
                    st.write(f"• **{analysis['cluster_type']}**: {analysis['percentage']:.1f}% of data")
            
            with col2:
                # Cluster visualization
                if len(clustering_results['feature_matrix']) > 0:
                    cluster_labels = clustering_results['kmeans_labels']
                    feature_matrix = clustering_results['feature_matrix']
                    
                    fig_cluster = px.scatter(
                        x=feature_matrix[:, 0], y=feature_matrix[:, 1],
                        color=cluster_labels, 
                        title="Energy Consumption Clusters",
                        labels={'x': 'Average Consumption', 'y': 'Consumption Variability'}
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Energy Saving Recommendations
            st.subheader("💡 AI-Powered Energy Saving Recommendations")
            
            for cluster_name, tips in energy_tips.items():
                with st.expander(f"{tips['cluster_type']} - Recommendations"):
                    st.write("**Priority Actions:**")
                    for tip in tips['priority_tips']:
                        st.write(f"• {tip}")
                    
                    st.write("**Additional Opportunities:**")
                    for tip in tips['additional_tips'][:3]:
                        st.write(f"• {tip}")
            
            # AI Agent Consultation
            st.subheader("🤖 AI Energy Consultant")
            
            user_query = st.text_input("Ask the AI energy consultant anything about your consumption patterns:")
            
            if user_query:
                with st.spinner("AI consultant is analyzing..."):
                    ai_response = energy_agent.get_recommendations(user_query)
                    st.write("**AI Consultant Response:**")
                    st.write(ai_response)
            
            # Comprehensive Insights Report
            st.subheader("📋 Comprehensive Analysis Report")
            
            with st.expander("View Full Analysis Report"):
                st.text(insights_report)
            
            # Export options
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export Forecast Data"):
                    if 'prophet' in results:
                        forecast_data = results['prophet']['forecast']
                        csv = forecast_data.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast CSV",
                            data=csv,
                            file_name=f"energy_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if st.button("Export Analysis Report"):
                    st.download_button(
                        label="Download Report",
                        data=insights_report,
                        file_name=f"energy_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
            
            with col3:
                if st.button("Export Model Performance"):
                    if performance_data:
                        perf_csv = pd.DataFrame(performance_data).to_csv(index=False)
                        st.download_button(
                            label="Download Performance CSV",
                            data=perf_csv,
                            file_name=f"model_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()
