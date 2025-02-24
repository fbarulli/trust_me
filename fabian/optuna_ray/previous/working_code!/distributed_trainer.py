import logging
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import traceback
from typing import Any, Dict, Optional
from wand_callback import WandbCallback
from universal_tracking import UniversalTrackingCallback
from datetime import datetime
import wandb

logger = logging.getLogger(__name__)

class RayTuneTrainer:
    def __init__(self, config, data_module, model_class, run_name=None, use_tune=True, base_path=None):
        self.config = config
        self.data_module = data_module
        self.model_class = model_class
        self.use_tune = use_tune
        self.run_name = run_name
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.project_root = self.base_path / "results"
        self.ray_results_dir = self.project_root / "ray_results"
        
        # Create directories
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.ray_results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RayTuneTrainer with base_path: {self.base_path}")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Ray results directory: {self.ray_results_dir}")

    def setup_callbacks(self, config):
        callbacks = []

        # Add tracking callback
        tracking_callback = UniversalTrackingCallback(
            framework='ray',
            results_dir=str(self.ray_results_dir),
            study_name=self.run_name
        )
        callbacks.append(tracking_callback)

        # Add early stopping callback if configured
        if config.get('training', {}).get('early_stopping_patience'):
            early_stopping = EarlyStopping(
                monitor=config['validation']['monitor_metric'],
                mode=config['validation']['monitor_mode'],
                patience=config['training']['early_stopping_patience'],
                verbose=True
            )
            callbacks.append(early_stopping)

        # Enhanced WandB callback setup
        if config.get('wandb'):
            wandb_callback = WandbCallback(
                project_name=config['wandb'].get('project_name', 'ray_tune'),
                run_name=f"{self.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,  # Pass the full config
                dir=str(self.ray_results_dir),  # Set the logging directory
                tags=['ray_tune', config.get('wandb', {}).get('tags', [])],
                group=config.get('wandb', {}).get('group', 'ray_experiments')
            )
            callbacks.append(wandb_callback)

        return callbacks

    def tune_function(self, config):
        try:
            logger.info("Starting tune_function")
            logger.info("="*50)
            logger.info(f"Config in tune_function: {config}")
            
            # Update config with trial-specific parameters
            updated_config = self.config.copy()
            updated_config.update(config)
            
            logger.info(f"Updated config after merge: {updated_config}")
            logger.info("="*50)
            
            # Update data module with new config
            self.data_module = self.data_module.__class__(
                config=updated_config,
                base_path=self.base_path
            )

            # Initialize model
            model = self.model_class(updated_config)
            
            # Initialize WandB logger
            wandb_logger = pl.loggers.WandbLogger(
                project=updated_config['wandb'].get('project_name', 'ray_tune'),
                name=f"{self.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=updated_config,
                save_dir=str(self.ray_results_dir)
            )

            # Initialize trainer with WandB logger
            trainer = pl.Trainer(
                max_epochs=self.config['scheduler']['max_t'],
                accelerator=self.config['accelerator'],
                devices=self.config['devices'],
                callbacks=self.setup_callbacks(updated_config),
                accumulate_grad_batches=self.config['training']['gradient_accumulation_steps'],
                precision=self.config['training']['precision'],
                enable_checkpointing=False,
                logger=wandb_logger
            )

            # Fit the model
            trainer.fit(model, self.data_module)

        except Exception as e:
            logger.error(f"Error in tune_function: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        return None

    def train_with_ray(self, search_space, num_samples=None):
        if not self.use_tune:
            raise ValueError("Ray Tune is disabled, cannot proceed with this configuration.")
        
        try:
            if not search_space:
                logger.warning("Search space is empty. Running with default config.")
                self.tune_function(self.config)
                return

            # Setup Ray Tune reporter
            reporter = CLIReporter(
                metric_columns=[
                    self.config['validation']['monitor_metric'],
                    "training_iteration"
                ]
            )

            # Setup ASHA scheduler
            scheduler = ASHAScheduler(
                metric=self.config['scheduler']['metric'],
                mode=self.config['scheduler']['mode'],
                max_t=self.config['scheduler']['max_t'],
                grace_period=self.config['scheduler']['grace_period'],
                reduction_factor=self.config['scheduler']['reduction_factor']
            )

            # Prepare tune config with base path
            tune_config = {
                "config": self.config,
                "base_path": str(self.base_path),
                **search_space
            }

            # Run optimization
            analysis = tune.run(
                lambda config: self.tune_function(config),  # Use lambda to properly bind the method
                config=tune_config,
                num_samples=num_samples or self.config['training']['num_samples'],
                scheduler=scheduler,
                progress_reporter=reporter,
                storage_path=str(self.ray_results_dir),
                name=self.run_name,
                resources_per_trial={
                    "cpu": self.config['training']['cpus_per_trial'],
                    "gpu": self.config['training']['gpus_per_trial']
                }
            )

            # Get best trial
            best_trial = analysis.get_best_trial(
                self.config['validation']['monitor_metric'],
                self.config['validation']['monitor_mode'],
                "last"
            )
            
            if best_trial:
                best_config = best_trial.config
                logger.info(f"Best trial config: {best_config}")
                return best_config
            
            return None

        except Exception as e:
            logger.error(f"Error in train_with_ray: {str(e)}")
            logger.error(traceback.format_exc())
            return None
