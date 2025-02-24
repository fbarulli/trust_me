import logging
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm
from advanced_dataset import AdvancedDataset
from config_utils import load_config

logger = logging.getLogger(__name__)

class AdvancedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", config=None, base_path=None):
        super().__init__()
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.data_dir = self.base_path / data_dir
        self.config = config
        
        # New directory structure
        self.training_data_dir = self.data_dir / "training_data"
        self.results_dir = self.base_path / "results"
        
        # Initialize attributes
        self.tokenizer_name = None
        self.max_length = None
        self.batch_size = None
        self.num_workers = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Create directories
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized with base_path: {self.base_path}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Training data directory: {self.training_data_dir}")
        logger.info(f"Results directory: {self.results_dir}")

    def update_batch_size(self, batch_size):
        """Update batch size (called by Ray Tune)"""
        self.batch_size = batch_size
        logger.info(f"Updated batch size to {batch_size}")

    def prepare_data(self):
        """Prepare data - download, tokenize, etc."""
        logger.info("Preparing data...")

        # Get configuration
        data_config = self.config['data_module']
        self.tokenizer_name = data_config['model_name'] 
        self.max_length = data_config['max_length'] 
        self.batch_size = data_config['batch_size']
        self.num_workers = data_config['num_workers']

        # Set up model cache in results directory
        model_cache_dir = self.results_dir / "model_cache"
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            tokenizer_cache_path = model_cache_dir / self.tokenizer_name
            if (tokenizer_cache_path).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name,
                    cache_dir=str(model_cache_dir)
                )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

        # Process data if needed
        if not self._check_processed_data():
            self._process_and_save_data()

    def setup(self, stage=None):
        """Setup datasets using training_data directory"""
        logger.info(f"Setting up data for stage: {stage}")

        try:
            if stage in ('fit', None):
                train_df = pd.read_csv(self.training_data_dir / "train_processed.csv")
                val_df = pd.read_csv(self.training_data_dir / "val_processed.csv")

                self.train_dataset = AdvancedDataset(
                    train_df['text'].tolist(),
                    train_df['label'].tolist(),
                    self.tokenizer,
                    self.max_length
                )

                self.val_dataset = AdvancedDataset(
                    val_df['text'].tolist(),
                    val_df['label'].tolist(),
                    self.tokenizer,
                    self.max_length
                )

            if stage in ('test', None):
                test_df = pd.read_csv(self.training_data_dir / "test_processed.csv")
                self.test_dataset = AdvancedDataset(
                    test_df['text'].tolist(),
                    test_df['label'].tolist(),
                    self.tokenizer,
                    self.max_length
                )

        except Exception as e:
            logger.error(f"Error in setup: {str(e)}")
            raise

    def _create_dataloader(self, dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True
        ) if dataset else None

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)

    def _check_processed_data(self):
        return all(
            (self.training_data_dir / f"{split}_processed.csv").exists()
            for split in ['train', 'val', 'test']
        )

    def _process_and_save_data(self):
        logger.info("Processing data...")
        try:
            # Construct absolute path to data file
            data_file = self.base_path / self.config['data_module']['data_path']
            
            # Enhanced logging for debugging
            logger.info(f"Base path: {self.base_path}")
            logger.info(f"Config data path: {self.config['data_module']['data_path']}")
            logger.info(f"Full data path: {data_file}")
            logger.info(f"File exists: {data_file.exists()}")

            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found at: {data_file}")

            df = pd.read_csv(data_file)
            df["label"] = df[self.config.get('data_module', {}).get('rating_column', 'rating')] - 1

            # Split and save data to training_data directory
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df[self.config.get('data_module', {}).get('text_column', 'text')].tolist(),
                df["label"].tolist(),
                test_size=0.2,
                random_state=42
            )

            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, random_state=42
            )

            for split_name, texts, labels in [
                ('train', train_texts, train_labels),
                ('val', val_texts, val_labels),
                ('test', test_texts, test_labels)
            ]:
                df_split = pd.DataFrame({'text': texts, 'label': labels})
                output_path = self.training_data_dir / f"{split_name}_processed.csv"
                df_split.to_csv(output_path, index=False)
                logger.info(f"Saved {split_name} split to {output_path}")

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
