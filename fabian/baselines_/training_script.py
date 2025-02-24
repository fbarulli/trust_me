# training_script.py
import os
import logging
import pytorch_lightning as pl
import pandas as pd
from data_module import AdvancedDataModule
from simple_model import SimpleTransformerClassifier
from utils import load_config, create_directories, save_training_plot, handle_tensorflow_warnings
from ray_trainer import RayTrainer
import traceback

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting main function")
    try:
        # Load configuration and setup
        config = load_config()
        logger.info("Configuration loaded successfully.")
        create_directories(config)
        logger.info("Directories created/verified.")
        handle_tensorflow_warnings()

        # Create directories variables
        results_dir = config['directories']['results_dir']
        trained_models_dir = config['directories']['trained_models_dir']
        plots_dir = config['directories']['plots_dir']

        # Initialize Ray Trainer
        trainer = RayTrainer(
            config=config,
            data_module_class=AdvancedDataModule,
            model_class=SimpleTransformerClassifier,
            run_name="concurrent_training",
            base_path=os.getcwd()
        )
        
        # Train all models concurrently
        results = trainer.train_all_models()
        
        # Process results and generate plots
        for result in results:
            if result["success"]:
                model_name = result["model_name"]
                # Generate plots if history exists
                if "history_path" in result:
                    try:
                        plots_model_dir = os.path.join(plots_dir, model_name)
                        os.makedirs(plots_model_dir, exist_ok=True)
                        history_df = pd.read_csv(result["history_path"])
                        save_training_plot(history_df, model_name, "loss", plots_model_dir)
                        save_training_plot(history_df, model_name, "acc", plots_model_dir)
                        logger.info(f"Training plots saved for {model_name}")
                    except Exception as e:
                        logger.error(f"Error generating plots for {model_name}: {e}")
                        logger.error(traceback.format_exc())

            else:
                logger.error(f"Training failed for {result['model_name']}: {result['error']}")

    except Exception as e:
        logger.critical(f"Fatal error in main function: {e}")
        logger.critical(traceback.format_exc())
        print(f"FATAL ERROR: {e}. See logs for details.")

    finally:
        logger.info("="*50)
        logger.info("Training script completed.")
        logger.info(f"Results are saved in the '{results_dir}' directory.")
        logger.info("="*50)

if __name__ == "__main__":
    main()