# gpu_tracker.py
import torch
import logging
import json
import os
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class GPUTracker:
    def __init__(self, output_dir="gpu_tracking"):
        self.history = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create CSV files for different metrics
        self.memory_file = os.path.join(output_dir, "memory_tracking.csv")
        self.gradients_file = os.path.join(output_dir, "gradient_tracking.csv")
        self.values_file = os.path.join(output_dir, "value_tracking.csv")
        
        # Initialize CSV headers
        pd.DataFrame(columns=['timestamp', 'tag', 'allocated_mb', 'reserved_mb', 'max_allocated_mb']).to_csv(self.memory_file, index=False)
        pd.DataFrame(columns=['timestamp', 'tag', 'param_name', 'mean', 'std', 'max', 'min', 'norm']).to_csv(self.gradients_file, index=False)
        pd.DataFrame(columns=['timestamp', 'tag', 'var_name', 'shape', 'mean', 'std', 'max', 'min']).to_csv(self.values_file, index=False)
        
    def track_step(self, tag: str, model: Optional[torch.nn.Module] = None, local_vars: Optional[Dict] = None):
        """Track GPU usage and values at a given step."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # Track GPU Memory
        gpu_stats = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
        }
        
        # Save memory stats
        pd.DataFrame([{
            'timestamp': timestamp,
            'tag': tag,
            'allocated_mb': gpu_stats['allocated'],
            'reserved_mb': gpu_stats['reserved'],
            'max_allocated_mb': gpu_stats['max_allocated']
        }]).to_csv(self.memory_file, mode='a', header=False, index=False)
        
        # Track model gradients if provided
        if model is not None:
            grad_records = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_stats = {
                        'timestamp': timestamp,
                        'tag': tag,
                        'param_name': name,
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item(),
                        'max': param.grad.max().item(),
                        'min': param.grad.min().item(),
                        'norm': param.grad.norm().item()
                    }
                    grad_records.append(grad_stats)
            
            if grad_records:
                pd.DataFrame(grad_records).to_csv(self.gradients_file, mode='a', header=False, index=False)
        
        # Track specific variables if provided
        if local_vars is not None:
            var_records = []
            for name, var in local_vars.items():
                if torch.is_tensor(var):
                    var_stats = {
                        'timestamp': timestamp,
                        'tag': tag,
                        'var_name': name,
                        'shape': str(var.shape),
                        'mean': var.mean().item(),
                        'std': var.std().item(),
                        'max': var.max().item(),
                        'min': var.min().item()
                    }
                    var_records.append(var_stats)
            
            if var_records:
                pd.DataFrame(var_records).to_csv(self.values_file, mode='a', header=False, index=False)
        
        # Clear memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_report(self):
        """Generate a summary report of the tracking data."""
        # Memory usage report
        memory_df = pd.read_csv(self.memory_file)
        gradients_df = pd.read_csv(self.gradients_file)
        values_df = pd.read_csv(self.values_file)
        
        report = {
            'memory': {
                'peak_usage_mb': memory_df['max_allocated_mb'].max(),
                'mean_usage_mb': memory_df['allocated_mb'].mean(),
                'by_step': memory_df.groupby('tag')['allocated_mb'].mean().to_dict()
            },
            'gradients': {
                'largest_gradient': gradients_df.loc[gradients_df['max'].idxmax()].to_dict(),
                'mean_gradient_norm': gradients_df['norm'].mean(),
                'params_with_large_grads': gradients_df[gradients_df['max'].abs() > 10]['param_name'].unique().tolist()
            },
            'values': {
                'largest_value': values_df.loc[values_df['max'].idxmax()].to_dict(),
                'variables_with_large_values': values_df[values_df['max'].abs() > 100]['var_name'].unique().tolist()
            }
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, "tracking_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self._generate_plots()
        
        return report
    
    def _generate_plots(self):
        """Generate visualization plots of the tracking data."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Memory usage over time
            plt.figure(figsize=(12, 6))
            memory_df = pd.read_csv(self.memory_file)
            plt.plot(memory_df['allocated_mb'], label='Allocated')
            plt.plot(memory_df['reserved_mb'], label='Reserved')
            plt.title('GPU Memory Usage Over Time')
            plt.xlabel('Step')
            plt.ylabel('Memory (MB)')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'memory_usage.png'))
            plt.close()
            
            # Gradient norms distribution
            plt.figure(figsize=(12, 6))
            gradients_df = pd.read_csv(self.gradients_file)
            sns.histplot(data=gradients_df, x='norm', bins=50)
            plt.title('Distribution of Gradient Norms')
            plt.xlabel('Gradient Norm')
            plt.savefig(os.path.join(self.output_dir, 'gradient_norms.png'))
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib and/or seaborn not available. Skipping plot generation.")