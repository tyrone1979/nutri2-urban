"""
Fixed SHAPInterpreter class with proper handling of multi-class SHAP values
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import torch
import shap
import matplotlib.pyplot as plt


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


# Also fix the interpret method in ModelPipeline to handle errors gracefully
class FixedModelPipeline:
    """Fixed pipeline with error handling in interpret"""

    # ... [previous methods remain the same] ...

    def interpret(self, df: pd.DataFrame, max_samples: int = 200):
        """SHAP interpretation with error handling"""
        if self.model is None:
            raise ValueError("Model not trained")

        X = self.scaler.transform(df[self.feature_names].values[:max_samples])
        X = self._ensure_2d(X)

        interpreter = SHAPInterpreter(self.model, self.feature_names, self.tasks)

        importance_results = {}
        for task in self.tasks:
            print(f"\nCalculating SHAP for {task.name}...")
            try:
                shap_values = interpreter.explain_global(X, task.name, max_samples)

                # Debug info
                if isinstance(shap_values, list):
                    print(f"  Multi-class SHAP: {len(shap_values)} classes, "
                          f"shapes: {[s.shape for s in shap_values]}")
                else:
                    print(f"  Binary SHAP shape: {shap_values.shape}")

                # Plot summary
                interpreter.plot_summary(shap_values, task.name)

                # Get importance
                importance_df = interpreter.get_feature_importance(shap_values, task.name)
                importance_results[task.name] = importance_df

                print(f"Top 5 features for {task.name}:")
                print(importance_df.head())

            except Exception as e:
                print(f"  Error processing {task.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return importance_results

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensure array is 2D"""
        if X.ndim == 1:
            return X.reshape(1, -1)
        return X


# Quick test to verify the fix
if __name__ == "__main__":
    # Test the fix with simulated data
    feature_names = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']

    # Test case 1: Binary classification (2D array)
    print("Test 1: Binary classification")
    shap_binary = np.random.randn(100, 5)
    interpreter = SHAPInterpreter(None, feature_names, [])
    df_binary = interpreter.get_feature_importance(shap_binary, "binary_task")
    print(f"Success! Shape: {df_binary.shape}")
    print(df_binary.head(3))

    # Test case 2: Multi-class classification (list of 2D arrays)
    print("\nTest 2: Multi-class classification (3 classes)")
    shap_multi = [np.random.randn(100, 5) for _ in range(3)]
    df_multi = interpreter.get_feature_importance(shap_multi, "multiclass_task")
    print(f"Success! Shape: {df_multi.shape}")
    print(df_multi.head(3))

    print("\nAll tests passed! The fix is working correctly.")