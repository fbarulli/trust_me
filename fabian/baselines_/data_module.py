# data_module.py
import logging
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from dataset import AdvancedDataset
from utils import load_config
import traceback
from transformers.tokenization_utils_base import BatchEncoding # Import BatchEncoding


logger = logging.getLogger(__name__)

class AdvancedDataModule(pl.LightningDataModule):
    def __init__(self, model_name, csv_name="so_many_rev.csv", batch_size=32, processed_data_dir="results/processed_data", cached_model_dir="results/cached_models"):
        super().__init__()
        logger.info(f"Starting AdvancedDataModule initialization for model: {model_name}")
        try:
            self.model_name = model_name
            self.csv_name = csv_name
            self.batch_size = batch_size
            self.processed_data_dir = processed_data_dir
            self.cached_model_dir = cached_model_dir
            self.tokenizer = None
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            self.max_length = 128 # You can configure this in config.yaml if needed
            logger.info(f"AdvancedDataModule initialized with model: {model_name}, batch_size: {batch_size}")
        except Exception as e:
            logger.error(f"Error during AdvancedDataModule initialization for {model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting AdvancedDataModule initialization for model: {model_name}")


    def prepare_data(self):
        """Downloads tokenizer and processes data if not already processed with logging."""
        logger.info(f"Starting prepare_data for model: {self.model_name}")
        try:
            tokenizer_path = os.path.join(self.cached_model_dir, self.model_name)
            os.makedirs(tokenizer_path, exist_ok=True)
            tokenizer_exists = os.path.exists(os.path.join(tokenizer_path, 'tokenizer_config.json')) # Basic check
            if tokenizer_exists:
                logger.info(f"Tokenizer found locally at {tokenizer_path}, loading...")
            else:
                logger.info(f"Tokenizer not found locally, downloading {self.model_name} tokenizer to {tokenizer_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=tokenizer_path)
            logger.info(f"Tokenizer for {self.model_name} loaded successfully.")

            processed_filepath = os.path.join(self.processed_data_dir, self.model_name, "processed_data.pt") # corrected path
            #DEBUG - always process data
            logger.info(f"Pre-processed data not found at {processed_filepath}, starting data processing.")
            self._process_data()
            # else:
            #     logger.info(f"Pre-processed data found at {processed_filepath}, will load in setup().")
            print(f"processed_filepath SAVE: {processed_filepath}") # SAVE PATH
        except Exception as e:
            logger.error(f"Error in prepare_data for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting prepare_data for model: {self.model_name}")



    def setup(self, stage=None):
        """Loads processed data and creates datasets with logging."""
        logger.info(f"Starting setup for stage: {stage}, model: {self.model_name}")
        try:
            processed_filepath = os.path.join(self.processed_data_dir, self.model_name, "processed_data.pt") # corrected path
            logger.info(f"Loading processed data from: {processed_filepath}")
            processed_data = torch.load(processed_filepath, weights_only=False) # weights_only=True is now safe because of allowlist
            # Access elements using tuple indices (0, 1, 2, 3, 4, 5)
            self.train_encodings, self.val_encodings, self.test_encodings, self.train_labels, self.val_labels, self.test_labels = processed_data
            print(f"processed_filepath LOAD: {processed_filepath}") # LOAD PATH

            self.train_dataset = AdvancedDataset(self.train_encodings, self.train_labels)
            self.val_dataset = AdvancedDataset(self.val_encodings, self.val_labels)
            self.test_dataset = AdvancedDataset(self.test_encodings, self.test_labels)
            logger.info(f"Datasets created: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}.")

        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError in setup for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        except EOFError as e: # ADDED EOFError handling
            logger.error(f"EOFError in setup (possibly corrupted file) for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise            
        except Exception as e:
            logger.error(f"Error in setup for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting setup for model: {self.model_name}, stage: {stage}")


    def train_dataloader(self):
        logger.info(f"train_dataloader requested for model: {self.model_name}")
        try:
            dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            logger.info(f"Train dataloader created with batch size: {self.batch_size}, shuffle=True, num_workers=4.")
            return dataloader
        except Exception as e:
            logger.error(f"Error creating train_dataloader for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting train_dataloader for model: {self.model_name}")


    def val_dataloader(self):
        logger.info(f"val_dataloader requested for model: {self.model_name}")
        try:
            dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
            logger.info(f"Validation dataloader created with batch size: {self.batch_size}, num_workers=4.")
            return dataloader
        except Exception as e:
            logger.error(f"Error creating val_dataloader for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting val_dataloader for model: {self.model_name}")


    def test_dataloader(self):
        logger.info(f"test_dataloader requested for model: {self.model_name}")
        try:
            dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
            logger.info(f"Test dataloader created with batch size: {self.batch_size}, num_workers=4.")
            return dataloader
        except Exception as e:
            logger.error(f"Error creating test_dataloader for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting test_dataloader for model: {self.model_name}")


    def _process_data(self):
        """Reads CSV, splits data, tokenizes, and saves processed data with detailed logging."""
        logger.info(f"Starting _process_data for model: {self.model_name}")
        try:
            df = pd.read_csv(self.csv_name)
            logger.info(f"CSV file read successfully, shape: {df.shape}") # Modified logging

            # Add data validation
            required_columns = {'rating', 'text'} # Set of required columns
            missing_columns = required_columns - set(df.columns) # Find missing columns
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}") # Raise error if missing columns

            # Add data cleaning
            df['text'] = df['text'].astype(str).apply(lambda x: x.strip()) # Convert 'text' column to string, remove whitespace
            df = df.dropna(subset=['text', 'rating']) # Drop rows with NaN in 'text' or 'rating'

            # Validate ratings
            valid_ratings = set(range(1, 6))  # Assuming 1-5 rating scale
            invalid_ratings = set(df['rating']) - valid_ratings # Find invalid ratings
            if invalid_ratings:
                raise ValueError(f"Invalid ratings found: {invalid_ratings}") # Raise error if invalid ratings

            df["label"] = df['rating'] - 1  # Assuming 'rating' column and labels are 0-indexed

            logger.info("Splitting data into train, val, test sets (stratified).")
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label'].tolist()
            )
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
            )
            logger.info(f"Data split complete. Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)} samples.")

            logger.info(f"Tokenizing data using {self.model_name} tokenizer...")
            train_encodings = self.tokenizer(train_texts, truncation=True, padding='max_length', max_length=self.max_length)
            val_encodings = self.tokenizer(val_texts, truncation=True, padding='max_length', max_length=self.max_length)
            test_encodings = self.tokenizer(test_texts, truncation=True, padding='max_length', max_length=self.max_length)
            logger.info("Tokenization complete.")

            # Create the directory if it doesn't exist
            processed_dir = os.path.join(self.processed_data_dir, self.model_name)
            os.makedirs(processed_dir, exist_ok=True)
            logger.info(f"Created directory: {processed_dir}")

            # Save the processed data
            processed_filepath = os.path.join(processed_dir, "processed_data.pt")
            torch.save((train_encodings, val_encodings, test_encodings, train_labels, val_labels, test_labels), processed_filepath)
            logger.info(f"Processed data saved successfully to: {processed_filepath}")

        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError in _process_data for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        except ValueError as e:
            logger.error(f"ValueError in _process_data for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _process_data for {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting _process_data for model: {self.model_name}")