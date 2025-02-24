# main.py
import logging
import os
from pathlib import Path
import torch
from config_utils import load_config, setup_directories, save_config, create_search_space, get_project_root, generate_wandb_names
from advanced_data_module import AdvancedDataModule
from optimized_model import OptimizedModel
from meta_learning_db import MetaLearningDatabase
from advanced_search_space import AdvancedSearchSpace
from pytorch_lightning import seed_everything

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get project root
    project_root = Path(get_project_root())
    logger.info(f"Project root: {project_root}")

    # Load and setup configuration
    config = load_config()

    # --- CHECK FOR ESSENTIAL CONFIG SECTIONS ---
    required_sections = ['directories', 'data_module', 'model_name', 'num_labels', 
                        'scheduler', 'search_space', 'training', 'validation', 
                        'wandb', 'advanced_search']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required config section '{section}' is missing in config.yaml")

    seed = config['seed']
    seed_everything(seed)

    # Set up directories and save config
    results_dir, ray_results_dir, model_cache_dir = setup_directories(config)
    save_config(config)

    # Generate wandb names
    project_name, run_name = generate_wandb_names(config)

    # Data Module setup with project root path
    data_module = AdvancedDataModule(
        config=config,
        base_path=project_root  # Pass the project root as base_path
    )
    data_module.prepare_data()
    data_module.setup()

    # Model setup
    model_class = OptimizedModel

    # Meta-learning database and search space
    meta_db = MetaLearningDatabase(project_root=project_root)

    # Use previous trials for advanced search space
    previous_trials = []  # In a real implementation, load trials from Optuna
    advanced_search = AdvancedSearchSpace(previous_trials, config=config)
    advanced_search.analyze_parameter_importance()
    advanced_search.fit_gmm_models()

    # Ray Tune Training
    use_tune = config['use_tune']
    if use_tune:
        from distributed_trainer import RayTuneTrainer
        search_space = create_search_space(config)
        ray_trainer = RayTuneTrainer(
            config=config,
            data_module=data_module,
            model_class=model_class,
            run_name=run_name,
            use_tune=use_tune,
            base_path=project_root  # Pass the project root to the trainer
        )

        try:
            best_config = ray_trainer.train_with_ray(
                search_space=search_space, 
                num_samples=config['training']['num_samples']
            )
            logger.info("Running tune_function with Ray Tune")

            if best_config:
                config.update(best_config)
                save_config(config, backup=False)
                logger.info(f"Best config updated: {config}")
        except Exception as e:
            logger.error(f"Error during Ray Tune training: {str(e)}")
            raise
    else:
        raise ValueError("Ray tune is disabled, cannot run in this config")

    logger.info("Training completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise
