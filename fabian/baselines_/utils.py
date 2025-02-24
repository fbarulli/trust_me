# utils.py
import os
import yaml
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import torch  # Import torch
import warnings # ADDED warnings

logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Loads configuration from yaml file."""
    logger.info(f"Starting load_config from path: {config_path}")
    try:
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from: {config_path}")
        return config
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in load_config: {e}")
        logger.error(traceback.format_exc())
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAMLError in load_config: {e}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Unexpected error in load_config: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Exiting load_config")

def create_directories(config):
    """Creates necessary directories if they don't exist with logging."""
    logger.info("Starting create_directories")
    try:
        dirs = [
            config['directories']['results_dir'],
            config['directories']['cached_models_dir'],
            config['directories']['trained_models_dir'],
            config['directories']['processed_data_dir'],
            config['directories']['plots_dir']
        ]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                logger.info(f"Directory already exists: {dir_path}")
        logger.info("Successfully created/verified directories.")
    except Exception as e:
        logger.error(f"Error in create_directories: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Exiting create_directories")


def save_training_plot(history, model_name, metric, save_dir):
    """Generates and saves training/validation plots with logging."""
    logger.info(f"Starting save_training_plot for {model_name}, metric: {metric}, save_dir: {save_dir}")
    try:
        plt.figure(figsize=(10, 6))

        # Move tensor data to CPU before plotting
        train_metric_data = history[f"train_{metric}"].cpu() if isinstance(history[f"train_{metric}"][0], torch.Tensor) else history[f"train_{metric}"]
        val_metric_data = history[f"val_{metric}"].cpu() if isinstance(history[f"val_{metric}"][0], torch.Tensor) else history[f"val_{metric}"]


        sns.lineplot(x=history["epoch"], y=train_metric_data, label=f"Training {metric.capitalize()}") # Use CPU data
        sns.lineplot(x=history["epoch"], y=val_metric_data,   label=f"Validation {metric.capitalize()}") # Use CPU data


        plt.title(f"{model_name.upper()} - Training and Validation {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

        plot_filename = os.path.join(save_dir, f"{model_name}_train_val_{metric}.png")
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Saved plot to: {plot_filename}")
    except Exception as e:
        logger.error(f"Error in save_training_plot for {model_name}, metric: {metric}: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info(f"Exiting save_training_plot for {model_name}, metric: {metric}")

def handle_tensorflow_warnings(): # ADDED function
    """Handle TensorFlow optimization warnings"""
    warnings.filterwarnings('ignore', message='.*AVX2.*')
    warnings.filterwarnings('ignore', message='.*AVX512F.*')
    warnings.filterwarnings('ignore', message='.*FMA.*')