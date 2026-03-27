"""
Complete Pipeline with History and Metrics Usage
Feature Engineering -> Multi-Task Modeling -> SHAP Interpretation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            classification_report, confusion_matrix, roc_curve, auc,
                            precision_recall_curve, average_precision_score)

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TaskConfig:
    """Configuration for each prediction task"""
    name: str
    target_col: str
    task_type: str
    num_classes: int
    loss_weight: float = 1.0


class CHNSDataSet(Dataset):
    """PyTorch Dataset for CHNS data"""

    def __init__(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        self.X = torch.FloatTensor(X)
        self.y_dict = {k: torch.LongTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y_dict.items()}


class MultiTaskNN(nn.Module):
    """Multi-Task Neural Network"""

    def __init__(self, input_dim: int, tasks: List[TaskConfig],
                 shared_layers: List[int] = [128, 64],
                 task_specific_layers: List[int] = [32]):
        super(MultiTaskNN, self).__init__()

        self.tasks = tasks

        # Shared layers
        shared_modules = []
        prev_dim = input_dim
        for dim in shared_layers:
            shared_modules.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        self.shared_layers = nn.Sequential(*shared_modules)
        self.shared_dim = prev_dim

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            task_modules = []
            prev_task_dim = self.shared_dim
            for dim in task_specific_layers:
                task_modules.extend([
                    nn.Linear(prev_task_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_task_dim = dim

            output_dim = task.num_classes if task.task_type != 'regression' else 1
            task_modules.append(nn.Linear(prev_task_dim, output_dim))
            self.task_heads[task.name] = nn.Sequential(*task_modules)

    def forward(self, x):
        shared_rep = self.shared_layers(x)
        outputs = {}
        for task in self.tasks:
            outputs[task.name] = self.task_heads[task.name](shared_rep)
        return outputs


class MultiTaskTrainer:
    """Trainer for Multi-Task Learning"""

    def __init__(self, model: MultiTaskNN, tasks: List[TaskConfig],
                 learning_rate: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.tasks = tasks
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        self.criteria = {}
        for task in tasks:
            if task.task_type in ['binary', 'multiclass']:
                self.criteria[task.name] = nn.CrossEntropyLoss()
            else:
                self.criteria[task.name] = nn.MSELoss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {t.name: {'accuracy': [], 'f1': []} for t in tasks},
            'val_metrics': {t.name: {'accuracy': [], 'f1': []} for t in tasks}
        }

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = {t.name: [] for t in self.tasks}
        all_targets = {t.name: [] for t in self.tasks}

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)

            batch_loss = 0
            for task in self.tasks:
                task_out = outputs[task.name]
                task_target = batch_y[task.name].to(self.device)
                task_loss = self.criteria[task.name](task_out, task_target)
                batch_loss += task.loss_weight * task_loss

                preds = torch.argmax(task_out, dim=1).cpu().numpy()
                all_preds[task.name].extend(preds)
                all_targets[task.name].extend(task_target.cpu().numpy())

            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()

        # Calculate metrics
        metrics = {}
        for task in self.tasks:
            metrics[task.name] = {
                'accuracy': accuracy_score(all_targets[task.name], all_preds[task.name]),
                'f1': f1_score(all_targets[task.name], all_preds[task.name],
                              average='weighted', zero_division=0)
            }

        return total_loss / len(dataloader), metrics

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = {t.name: [] for t in self.tasks}
        all_targets = {t.name: [] for t in self.tasks}
        all_probs = {t.name: [] for t in self.tasks}

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)

                for task in self.tasks:
                    task_out = outputs[task.name]
                    task_target = batch_y[task.name].to(self.device)

                    task_loss = self.criteria[task.name](task_out, task_target)
                    total_loss += task_loss.item()

                    probs = torch.softmax(task_out, dim=1).cpu().numpy()
                    preds = torch.argmax(task_out, dim=1).cpu().numpy()

                    all_probs[task.name].extend(probs)
                    all_preds[task.name].extend(preds)
                    all_targets[task.name].extend(task_target.cpu().numpy())

        metrics = {}
        for task in self.tasks:
            metrics[task.name] = {
                'accuracy': accuracy_score(all_targets[task.name], all_preds[task.name]),
                'precision': precision_score(all_targets[task.name], all_preds[task.name],
                                           average='weighted', zero_division=0),
                'recall': recall_score(all_targets[task.name], all_preds[task.name],
                                     average='weighted', zero_division=0),
                'f1': f1_score(all_targets[task.name], all_preds[task.name],
                              average='weighted', zero_division=0),
                'predictions': np.array(all_preds[task.name]),
                'targets': np.array(all_targets[task.name]),
                'probabilities': np.array(all_probs[task.name])
            }

        return total_loss / len(dataloader), metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 50, patience: int = 5) -> Dict:
        """Train with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            for task in self.tasks:
                for metric in ['accuracy', 'f1']:
                    self.history['train_metrics'][task.name][metric].append(
                        train_metrics[task.name][metric]
                    )
                    self.history['val_metrics'][task.name][metric].append(
                        val_metrics[task.name][metric]
                    )

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                for task_name in [t.name for t in self.tasks]:
                    print(f"  {task_name}: "
                          f"Train Acc={train_metrics[task_name]['accuracy']:.4f}, "
                          f"Val Acc={val_metrics[task_name]['accuracy']:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(self.best_state)
        return self.history


class ModelVisualizer:
    """Visualization tools for training history and metrics"""

    def __init__(self, save_dir: str = './figures'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: Dict, tasks: List[TaskConfig]):
        """Plot training and validation loss/metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Task-specific metrics
        for idx, task in enumerate(tasks):
            row = (idx + 1) // 2
            col = (idx + 1) % 2

            if row < 2 and col < 2:
                axes[row, col].plot(history['train_metrics'][task.name]['accuracy'],
                                   label='Train Acc', linestyle='--')
                axes[row, col].plot(history['val_metrics'][task.name]['accuracy'],
                                   label='Val Acc')
                axes[row, col].plot(history['train_metrics'][task.name]['f1'],
                                   label='Train F1', linestyle=':')
                axes[row, col].plot(history['val_metrics'][task.name]['f1'],
                                   label='Val F1')
                axes[row, col].set_title(f'{task.name} Metrics')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Score')
                axes[row, col].legend()
                axes[row, col].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"Saved training history to {self.save_dir / 'training_history.png'}")
        plt.show()
        return fig

    def plot_confusion_matrices(self, metrics: Dict, tasks: List[TaskConfig]):
        """Plot confusion matrices for all tasks"""
        n_tasks = len(tasks)
        fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 5))

        if n_tasks == 1:
            axes = [axes]

        for idx, task in enumerate(tasks):
            cm = confusion_matrix(metrics[task.name]['targets'],
                                 metrics[task.name]['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{task.name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrices to {self.save_dir / 'confusion_matrices.png'}")
        plt.show()
        return fig

    def plot_roc_curves(self, metrics: Dict, tasks: List[TaskConfig]):
        """Plot ROC curves for binary classification tasks"""
        fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 5))

        if len(tasks) == 1:
            axes = [axes]

        for idx, task in enumerate(tasks):
            if task.task_type == 'binary' and task.num_classes == 2:
                y_true = metrics[task.name]['targets']
                y_prob = metrics[task.name]['probabilities'][:, 1]

                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                axes[idx].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
                axes[idx].plot([0, 1], [0, 1], 'k--', label='Random')
                axes[idx].set_xlabel('False Positive Rate')
                axes[idx].set_ylabel('True Positive Rate')
                axes[idx].set_title(f'{task.name} ROC Curve')
                axes[idx].legend()
                axes[idx].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {self.save_dir / 'roc_curves.png'}")
        plt.show()
        return fig

    def generate_classification_reports(self, metrics: Dict, tasks: List[TaskConfig]) -> Dict:
        """Generate and save classification reports"""
        reports = {}

        for task in tasks:
            report = classification_report(
                metrics[task.name]['targets'],
                metrics[task.name]['predictions'],
                output_dict=True
            )
            reports[task.name] = report

            # Print report
            print(f"\n{'='*50}")
            print(f"Classification Report: {task.name}")
            print(f"{'='*50}")
            print(classification_report(
                metrics[task.name]['targets'],
                metrics[task.name]['predictions']
            ))

        # Save to JSON
        with open(self.save_dir / 'classification_reports.json', 'w') as f:
            json.dump(reports, f, indent=2)

        return reports


class SHAPInterpreter:
    """SHAP-based Model Interpretation - Fixed for multi-class"""

    def __init__(self, model, feature_names: List[str],
                 tasks: List, device: str = 'cpu'):
        self.model = model
        self.feature_names = feature_names
        self.tasks = tasks
        self.device = device
        self.explainers = {}

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensure array is 2D"""
        if X.ndim == 1:
            return X.reshape(1, -1)
        return X

    def create_explainer(self, background_data: np.ndarray, task_name: str):
        """Create SHAP KernelExplainer for a task"""
        self.model.eval()

        def model_forward(x):
            x = self._ensure_2d(x)
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                out = self.model(x_tensor)
                probs = torch.softmax(out[task_name], dim=1).cpu().numpy()
                return probs

        # Ensure background is 2D
        background_data = self._ensure_2d(background_data)
        explainer = shap.KernelExplainer(model_forward, background_data)
        self.explainers[task_name] = explainer
        return explainer

    def explain_global(self, X: np.ndarray, task_name: str,
                       max_samples: int = 100) -> Union[np.ndarray, List[np.ndarray]]:
        """Calculate SHAP values"""
        if task_name not in self.explainers:
            background = shap.sample(X, min(100, len(X)))
            self.create_explainer(background, task_name)

        X_sample = X[:max_samples] if len(X) > max_samples else X
        X_sample = self._ensure_2d(X_sample)

        shap_values = self.explainers[task_name].shap_values(X_sample)
        return shap_values

    def get_feature_importance(self, shap_values: Union[np.ndarray, List[np.ndarray]],
                               task_name: str) -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values
        Handles both binary (2D array) and multi-class (list of 2D arrays) cases
        """
        if isinstance(shap_values, list):
            # Multi-class case: list of 2D arrays, one per class
            # Each array shape: (n_samples, n_features)
            importance_per_class = []
            for class_shap in shap_values:
                class_shap = self._ensure_2d(class_shap)
                # Mean absolute value across samples for this class
                class_importance = np.abs(class_shap).mean(axis=0)
                importance_per_class.append(class_importance)

            # Average across all classes
            importance = np.mean(importance_per_class, axis=0)

        elif isinstance(shap_values, np.ndarray):
            # Binary case: single 2D array (n_samples, n_features)
            shap_values = self._ensure_2d(shap_values)
            importance = np.abs(shap_values).mean(axis=0)

        else:
            raise ValueError(f"Unexpected SHAP values type: {type(shap_values)}")

        # Ensure importance is 1D
        importance = np.asarray(importance).flatten()

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_summary(self, shap_values: Union[np.ndarray, List[np.ndarray]],
                     task_name: str, max_display: int = 10):
        """Plot SHAP summary with proper handling for multi-class"""
        plt.figure(figsize=(10, 8))

        # For multi-class, use the mean absolute SHAP values across all classes
        if isinstance(shap_values, list):
            # Calculate mean magnitude across classes
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            # Use first class's actual values for direction, scaled by importance
            plot_values = shap_values[0] * (mean_shap / (np.abs(shap_values[0]) + 1e-10))
        else:
            plot_values = shap_values

        plot_values = self._ensure_2d(plot_values)

        shap.summary_plot(
            plot_values,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        plt.title(f"SHAP Feature Importance - {task_name}")
        plt.tight_layout()
        plt.savefig(f'./figures/shap_summary_{task_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        return plt.gcf()

    def plot_waterfall(self, shap_values: Union[np.ndarray, List[np.ndarray]],
                       sample_idx: int = 0, task_name: str = ""):
        """Plot waterfall for a single prediction"""
        plt.figure(figsize=(12, 6))

        if isinstance(shap_values, list):
            # For multi-class, plot the predicted class
            values = shap_values[0][sample_idx]  # Simplified: uses first class
        else:
            values = shap_values[sample_idx]

        # Create explanation object for plotting
        expected_value = 0  # KernelExplainer doesn't provide expected value directly

        shap.waterfall_plot(shap.Explanation(
            values=values,
            base_values=expected_value,
            data=None,
            feature_names=self.feature_names
        ), show=False)

        plt.title(f"SHAP Waterfall - {task_name} (Sample {sample_idx})")
        plt.tight_layout()
        plt.savefig(f'./figures/shap_waterfall_{task_name}_{sample_idx}.png', dpi=300)
        plt.show()
        return plt.gcf()



class ModelPipeline:
    """Complete Pipeline"""

    def __init__(self, tasks: List[TaskConfig], scaling_method: str = 'standard'):
        self.tasks = tasks
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.model = None
        self.trainer = None
        self.visualizer = ModelVisualizer()
        self.history = None
        self.metrics = None
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str],
                     test_size: float = 0.2, batch_size: int = 256):
        """Prepare data loaders"""
        # Scale features
        X = self.scaler.fit_transform(df[feature_cols])
        self.feature_names = feature_cols

        # Prepare targets
        y_dict = {}
        for task in self.tasks:
            le = LabelEncoder()
            y_dict[task.name] = le.fit_transform(df[task.target_col])

        # Split data
        indices = np.arange(len(X))
        train_idx, temp_idx = train_test_split(
            indices, test_size=test_size, random_state=42,
            stratify=y_dict[self.tasks[0].name]
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42,
            stratify=y_dict[self.tasks[0].name][temp_idx]
        )

        # Create datasets
        train_set = CHNSDataSet(X[train_idx], {k: v[train_idx] for k, v in y_dict.items()})
        val_set = CHNSDataSet(X[val_idx], {k: v[val_idx] for k, v in y_dict.items()})
        test_set = CHNSDataSet(X[test_idx], {k: v[test_idx] for k, v in y_dict.items()})

        # Create loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              shared_layers: List[int] = [128, 64],
              task_specific_layers: List[int] = [32],
              epochs: int = 50, learning_rate: float = 0.001) -> Dict:
        """Train model"""
        sample_X, _ = next(iter(train_loader))
        input_dim = sample_X.shape[1]

        self.model = MultiTaskNN(input_dim, self.tasks, shared_layers, task_specific_layers)
        self.trainer = MultiTaskTrainer(self.model, self.tasks, learning_rate)

        self.history = self.trainer.fit(train_loader, val_loader, epochs=epochs)
        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate and store metrics"""
        if self.trainer is None:
            raise ValueError("Model not trained")

        test_loss, self.metrics = self.trainer.evaluate(test_loader)

        print("\n" + "="*60)
        print("Final Test Set Performance")
        print("="*60)
        print(f"Test Loss: {test_loss:.4f}\n")

        for task_name, task_metrics in self.metrics.items():
            print(f"{task_name}:")
            print(f"  Accuracy:  {task_metrics['accuracy']:.4f}")
            print(f"  Precision: {task_metrics['precision']:.4f}")
            print(f"  Recall:    {task_metrics['recall']:.4f}")
            print(f"  F1-Score:  {task_metrics['f1']:.4f}")

        return self.metrics

    def visualize_results(self):
        """Generate all visualizations using history and metrics"""
        if self.history is None or self.metrics is None:
            raise ValueError("Must train and evaluate first")

        print("\nGenerating visualizations...")

        # Plot training history
        self.visualizer.plot_training_history(self.history, self.tasks)

        # Plot confusion matrices
        self.visualizer.plot_confusion_matrices(self.metrics, self.tasks)

        # Plot ROC curves
        self.visualizer.plot_roc_curves(self.metrics, self.tasks)

        # Generate classification reports
        reports = self.visualizer.generate_classification_reports(self.metrics, self.tasks)

        return reports

    def interpret(self, df: pd.DataFrame, max_samples: int = 200):
        """SHAP interpretation"""
        if self.model is None:
            raise ValueError("Model not trained")

        X = self.scaler.transform(df[self.feature_names].values[:max_samples])

        interpreter = SHAPInterpreter(self.model, self.feature_names, self.tasks)

        importance_results = {}
        for task in self.tasks:
            print(f"\nCalculating SHAP for {task.name}...")
            shap_values = interpreter.explain_global(X, task.name, max_samples)

            # Plot summary
            interpreter.plot_summary(shap_values, task.name)

            # Get importance dataframe
            importance_df = interpreter.get_feature_importance(shap_values, task.name)
            importance_results[task.name] = importance_df

            print(f"Top 5 features for {task.name}:")
            print(importance_df.head())

        return importance_results

    def save_results(self, output_dir: str = './results'):
        """Save history and metrics to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save history
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss']
        })
        history_df.to_csv(output_dir / 'training_history.csv', index=False)

        # Save metrics
        metrics_summary = {}
        for task_name, task_metrics in self.metrics.items():
            metrics_summary[task_name] = {
                'accuracy': float(task_metrics['accuracy']),
                'precision': float(task_metrics['precision']),
                'recall': float(task_metrics['recall']),
                'f1': float(task_metrics['f1'])
            }

        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        print(f"\nResults saved to {output_dir}")
        print(f"  - training_history.csv")
        print(f"  - test_metrics.json")


# ============================================
# Main Execution
# ============================================

def main():
    # Load cleaned data
    print("Loading data...")
    df = pd.read_csv('./data/c12diet_cleaned.csv')

    # Define tasks
    tasks = [
        TaskConfig(name='diet_pattern', target_col='diet_pattern',
                   task_type='multiclass', num_classes=3, loss_weight=1.0),
        TaskConfig(name='urban_rural', target_col='urban_rural',
                   task_type='binary', num_classes=2, loss_weight=1.0),
        TaskConfig(name='high_energy', target_col='high_energy',
                   task_type='binary', num_classes=2, loss_weight=0.8)
    ]

    # Initialize pipeline
    pipeline = ModelPipeline(tasks, scaling_method='standard')

    # Feature preparation
    print("\nPreparing features...")
    feature_cols = ['d3kcal', 'd3carbo', 'd3fat', 'd3protn',
                   'carb_ratio', 'fat_ratio', 'prot_ratio', 'wave', 't1']
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Prepare data
    train_loader, val_loader, test_loader, indices = pipeline.prepare_data(
        df, feature_cols, test_size=0.2, batch_size=256
    )

    # Train model
    print("\n" + "="*60)
    print("Training Multi-Task Model")
    print("="*60)
    history = pipeline.train(
        train_loader, val_loader,
        shared_layers=[128, 64],
        task_specific_layers=[32],
        epochs=50,
        learning_rate=0.001
    )

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    metrics = pipeline.evaluate(test_loader)

    # Visualize results (using history and metrics)
    reports = pipeline.visualize_results()

    # SHAP interpretation
    print("\n" + "="*60)
    print("SHAP Interpretation")
    print("="*60)
    importance_results = pipeline.interpret(df, max_samples=200)

    # Save all results
    pipeline.save_results('./results')

    print("\n" + "="*60)
    print("Pipeline Completed Successfully!")
    print("="*60)

    # Return all results for further analysis
    return {
        'pipeline': pipeline,
        'history': history,
        'metrics': metrics,
        'reports': reports,
        'importance': importance_results
    }


if __name__ == "__main__":
    results = main()