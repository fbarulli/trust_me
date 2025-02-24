# dataset.py
import torch
import logging
import traceback

logger = logging.getLogger(__name__)

class AdvancedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        logger.info("Starting AdvancedDataset initialization")
        try:
            self.encodings = encodings
            self.labels = labels
            logger.info(f"AdvancedDataset initialized with {len(labels)} samples.")
        except Exception as e:
            logger.error(f"Error during AdvancedDataset initialization: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info("Exiting AdvancedDataset initialization")

    def __getitem__(self, idx):
        try:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {e}")
            logger.error(traceback.format_exc())
            raise

    def __len__(self):
        try:
            length = len(self.labels)
            logger.debug(f"__len__ called, returning: {length}") # Debug level for less critical info
            return length
        except Exception as e:
            logger.error(f"Error in __len__: {e}")
            logger.error(traceback.format_exc())
            return 0