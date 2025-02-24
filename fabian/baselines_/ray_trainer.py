# ray_trainer.py
import logging
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import ray
import torch
import traceback
from typing import Dict
import pandas as pd
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

@ray.remote(num_gpus=0.25)  # Share GPU memory across workers
class RayTrainerWorker:
    """Worker class for training individual models"""
    def __init__(self, model_config: Dict, base_path: str):
        self.config = model_config
        self.base_path = Path(base_path)
        self.results_dir = self.base_path / "results"
        self.trained_models_dir = self.results_dir / "trained_models"
        self.plots_dir = self.results_dir / "plots"
        
        # Set GPU memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.25)
            torch.cuda.empty_cache()

    def setup_callbacks(self):
        callbacks = []
        
        # Early stopping callback with default values if not in config
        early_stopping_config = self.config.get('validation', {
            'monitor_metric': 'val_loss',
            'monitor_mode': 'min',
            'early_stopping_patience': 3
        })

        early_stopping = EarlyStopping(
            monitor=early_stopping_config.get('monitor_metric', 'val_loss'),
            patience=self.config.get('training', {}).get('early_stopping_patience', 3),
            mode=early_stopping_config.get('monitor_mode', 'min'),
            verbose=True
        )
        callbacks.append(early_stopping)

        # Clear checkpoint directory
        checkpoint_dir = self.trained_models_dir / self.config['model_name']
        if checkpoint_dir.exists():
            shutil.rmtree(str(checkpoint_dir))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint callback
        model_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
            monitor=early_stopping_config.get('monitor_metric', 'val_loss'),
            mode=early_stopping_config.get('monitor_mode', 'min'),
            save_top_k=1,
            verbose=True
        )
        callbacks.append(model_checkpoint)

        return callbacks

    def train(self, data_module_class, model_class):
        try:
            logger.info(f"Starting training process for model: {self.config['model_name']}")

            # Initialize data module with proper batch size access
            data_module = data_module_class(
                model_name=self.config['model_name'],
                csv_name=self.config['csv_name'],
                batch_size=self.config.get('batch_size', self.config.get('training', {}).get('batch_size', 32)),
                processed_data_dir=str(self.results_dir / "processed_data"),
                cached_model_dir=str(self.results_dir / "cached_models")
            )
            
            # Prepare and setup data
            data_module.prepare_data()
            data_module.setup()

            # Initialize model with proper learning rate access
            model = model_class(
                model_name=self.config['model_name'],
                learning_rate=self.config['learning_rate'],
                cached_model_dir=str(self.results_dir / "cached_models")
            )

            # Initialize trainer with merged config values
            trainer = pl.Trainer(
                max_epochs=self.config.get('training', {}).get('epochs', 3),
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=self.setup_callbacks(),
                precision=self.config.get('training', {}).get('precision', "16-mixed"),
                accumulate_grad_batches=self.config.get('training', {}).get('gradient_accumulation_steps', 1),
                enable_model_summary=False,
                enable_progress_bar=True,
                enable_checkpointing=True,
                logger=False,
                strategy="auto"
            )

            # Train and test
            trainer.fit(model, datamodule=data_module)
            test_results = trainer.test(model, datamodule=data_module)

            # Save history if available
            if hasattr(model, 'train_history') and model.train_history:
                history_df = pd.DataFrame(model.train_history)
                plots_model_dir = self.plots_dir / self.config['model_name']
                plots_model_dir.mkdir(parents=True, exist_ok=True)
                
                history_csv_path = plots_model_dir / f"{self.config['model_name']}_training_history.csv"
                history_df.to_csv(history_csv_path, index=False)

                return {
                    "model_name": self.config['model_name'],
                    "test_results": test_results,
                    "history_path": str(history_csv_path),
                    "success": True
                }

            return {
                "model_name": self.config['model_name'],
                "test_results": test_results,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error training model {self.config['model_name']}: {e}")
            logger.error(traceback.format_exc())
            return {
                "model_name": self.config['model_name'],
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False
            }

class RayTrainer:
    def __init__(self, config: Dict, data_module_class, model_class, run_name=None, base_path=None):
        self.config = config
        self.data_module_class = data_module_class
        self.model_class = model_class
        self.run_name = run_name
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.results_dir = self.base_path / "results"
        self.ray_results_dir = self.results_dir / "ray_results"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ray_results_dir.mkdir(parents=True, exist_ok=True)

    def train_all_models(self):
        """Train all models concurrently using Ray"""
        try:
            # Initialize Ray
            ray.init(ignore_reinit_error=True)
            
            # Create training tasks for each model
            training_tasks = []
            
            # Iterate through each model in model_names
            for model_name, batch_size in self.config['model_names'].items():
                # Create specific config for this model
                model_config = {
                    'model_name': model_name,
                    'batch_size': batch_size,
                    'learning_rate': self.config['learning_rates'].get(model_name, 2e-5),
                    'csv_name': self.config['csv_name'],
                    'training': self.config['training'],
                    'validation': self.config['validation'],
                    'directories': self.config['directories']
                }

                # Create a worker for each model
                worker = RayTrainerWorker.remote(
                    model_config=model_config,
                    base_path=str(self.base_path)
                )
                
                # Launch training task
                training_tasks.append(
                    worker.train.remote(self.data_module_class, self.model_class)
                )

            # Wait for all training tasks to complete
            results = ray.get(training_tasks)
            
            # Process results
            for result in results:
                if result["success"]:
                    logger.info(f"Training completed successfully for {result['model_name']}")
                    logger.info(f"Test results: {result['test_results']}")
                else:
                    logger.error(f"Training failed for {result['model_name']}")
                    logger.error(f"Error: {result['error']}")

            return results

        except Exception as e:
            logger.error(f"Error in train_all_models: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            ray.shutdown()
            logger.info("Ray shutdown completed")