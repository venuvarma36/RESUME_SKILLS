"""
ML Model Module for Resume Skill Recognition System
Trains and evaluates classification models for skill categorization.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_logger, config


logger = get_logger(__name__)


class SkillClassifier:
    """Multi-label classifier for skill categorization."""
    
    def __init__(self, model_type: str = None):
        """
        Initialize skill classifier.
        
        Args:
            model_type: Type of model ('svm', 'random_forest', 'logistic_regression')
        """
        if model_type is None:
            model_type = config.get('ml_model.model_type', 'svm')
        
        self.model_type = model_type
        self.test_size = config.get('ml_model.test_size', 0.2)
        self.random_state = config.get('ml_model.random_state', 42)
        
        # Initialize model
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training history
        self.training_history = {}
        self.is_trained = False
        
        logger.info("SkillClassifier initialized with model_type: %s", model_type)
    
    def _create_model(self):
        """
        Create ML model based on configuration.
        
        Returns:
            Initialized model
        """
        if self.model_type == 'svm':
            return SVC(
                kernel=config.get('ml_model.kernel', 'linear'),
                max_iter=config.get('ml_model.max_iterations', 1000),
                class_weight=config.get('ml_model.class_weight', 'balanced'),
                random_state=self.random_state,
                probability=True
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                class_weight=config.get('ml_model.class_weight', 'balanced'),
                random_state=self.random_state
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=config.get('ml_model.max_iterations', 1000),
                class_weight=config.get('ml_model.class_weight', 'balanced'),
                random_state=self.random_state
            )
        else:
            logger.warning("Unknown model type: %s, defaulting to SVM", self.model_type)
            return SVC(
                kernel='linear',
                random_state=self.random_state,
                probability=True
            )
    
    def train(self, X: np.ndarray, y: np.ndarray,
              validate: bool = True) -> Dict[str, float]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            validate: Whether to perform validation
            
        Returns:
            Training metrics
        """
        logger.info("Training %s classifier with %d samples", 
                   self.model_type, len(X))
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_encoded
            )
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        train_metrics = self._compute_metrics(y_train, train_pred, "train")
        
        if validate:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            val_metrics = self._compute_metrics(y_val, val_pred, "validation")
            
            metrics = {**train_metrics, **val_metrics}
        else:
            metrics = train_metrics
        
        # Cross-validation
        if validate:
            cv_folds = config.get('ml_model.cross_validation_folds', 5)
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=cv_folds, scoring='accuracy'
            )
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
            
            logger.info("Cross-validation accuracy: %.4f (+/- %.4f)",
                       cv_scores.mean(), cv_scores.std())
        
        self.training_history = metrics
        
        logger.info("Training complete. Train accuracy: %.4f",
                   metrics['train_accuracy'])
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            logger.warning("Model does not support probability prediction")
            return None
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                        prefix: str = "") -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Evaluation metrics and report
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model on %d samples", len(X))
        
        # Predict
        y_pred = self.predict(X)
        y_encoded = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        
        # Compute metrics
        metrics = self._compute_metrics(y_encoded, y_pred_encoded, "test")
        
        # Classification report
        report = classification_report(
            y_encoded, y_pred_encoded,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_encoded, y_pred_encoded)
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray,
                             save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix
            save_path: Path to save plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Confusion matrix saved to %s", save_path)
        
        return fig
    
    def save_model(self, file_path: str):
        """
        Save trained model to file.
        
        Args:
            file_path: Path to save model
        """
        if not self.is_trained:
            logger.warning("Model not trained, saving untrained model")
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved to %s", file_path)
    
    def load_model(self, file_path: str):
        """
        Load trained model from file.
        
        Args:
            file_path: Path to model file
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        logger.info("Model loaded from %s", file_path)
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance (if supported by model).
        
        Args:
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False)
        
        return df
