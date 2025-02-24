# simple_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint
import logging
import traceback
import os

logger = logging.getLogger(__name__)

class SimpleTransformerClassifier(pl.LightningModule):
    def __init__(self, model_name, learning_rate, cached_model_dir="results/cached_models"):
        super().__init__()
        logger.info(f"Starting SimpleTransformerClassifier initialization for model: {model_name}")
        try:
            self.model_name = model_name
            self.learning_rate = learning_rate
            self.cached_model_dir = cached_model_dir
            self.save_hyperparameters()

            config_path = os.path.join(self.cached_model_dir, self.model_name)
            os.makedirs(config_path, exist_ok=True)

            # Always download from hub first, then save to cache
            logger.info(f"Loading configuration for {self.model_name}")
            self.config = AutoConfig.from_pretrained(
                self.model_name,  # Use model name directly to download from hub
                num_labels=5
            )
            
            logger.info(f"Loading model {self.model_name} from Hugging Face hub")
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,  # Use model name directly to download from hub
                config=self.config
            )
            
            # Save the model to cache after loading
            logger.info(f"Saving model to cache directory: {config_path}")
            self.transformer_model.save_pretrained(config_path)
            
            self.num_classes = self.config.num_labels
            self.train_history = []
            self.val_history = []
            
            logger.info(f"Model initialized successfully with {self.num_classes} classes")
            
        except Exception as e:
            logger.error(f"Error during SimpleTransformerClassifier initialization for {model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting SimpleTransformerClassifier initialization for model: {model_name}")

    def _init_classifier_weights(self): # ADDED weight initialization function
        """Initialize classifier weights properly"""
        if hasattr(self.transformer_model, 'classifier'):
            # Initialize dense layer
            if hasattr(self.transformer_model.classifier, 'dense'):
                torch.nn.init.xavier_uniform_(self.transformer_model.classifier.dense.weight)
                if self.transformer_model.classifier.dense.bias is not None:
                    torch.nn.init.zeros_(self.transformer_model.classifier.dense.bias)
            
            # Initialize output projection
            if hasattr(self.transformer_model.classifier, 'out_proj'):
                torch.nn.init.xavier_uniform_(self.transformer_model.classifier.out_proj.weight)
                if self.transformer_model.classifier.out_proj.bias is not None:
                    torch.nn.init.zeros_(self.transformer_model.classifier.out_proj.bias)


    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        try:
            # Prepare input tuple for transformer_model.forward()
            transformer_args = (input_ids, attention_mask, labels) # Removed token_type_ids

            # Call transformer_model's forward method directly and wrap with checkpoint
            outputs = checkpoint(self._transformer_forward, *transformer_args, use_reentrant=False)  # Pass inputs as positional args and set use_reentrant
            return outputs
        except Exception as e:
            logger.error(f"Error in forward pass for model {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise

    def _transformer_forward(self, input_ids, attention_mask, labels=None): # Removed token_type_ids
    # Separate method to call transformer's forward, to be wrapped by checkpoint
        return self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # Removed token_type_ids


    def training_step(self, batch, batch_idx):
        try:
            outputs = self(**batch) # **batch unpacks input_ids, attention_mask, token_type_ids (if present), labels
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()
            grad_norm = self.compute_grad_norm() # Compute gradient norm

            # Log metrics with prog_bar=True to display in progress bar
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=True) # Log gradient norm

            return loss
        except Exception as e:
            logger.error(f"Error in training_step for model {self.model_name}, batch_idx {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            raise

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()

            # Log metrics with prog_bar=True to display in progress bar
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True) # No prog_bar on step for cleaner output
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True) # No prog_bar on step for cleaner output

            return {'val_loss': loss, 'val_acc': acc} # Return for epoch-end aggregation
        except Exception as e:
            logger.error(f"Error in validation_step for model {self.model_name}, batch_idx {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            return {'val_loss': torch.tensor(float('nan')), 'val_acc': torch.tensor(float('nan'))} # Return NaN values
    
    def test_step(self, batch, batch_idx): # ADDED test_step
        try:
            outputs = self(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()

            # Log test metrics
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

            return {'test_loss': loss, 'test_acc': acc} # For aggregation if needed
        except Exception as e:
            logger.error(f"Error in test_step for model {self.model_name}, batch_idx {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            return {'test_loss': torch.tensor(float('nan')), 'test_acc': torch.tensor(float('nan'))} # Return NaN values


    def on_validation_epoch_end(self):
        """Store metrics at the end of validation epoch with logging."""
        logger.info(f"Starting on_validation_epoch_end for epoch {self.current_epoch}, model: {self.model_name}")
        try:
            metrics = self.trainer.callback_metrics # Access logged metrics from trainer
            epoch = self.current_epoch
            train_loss = metrics.get('train_loss_epoch', float('nan'))
            train_acc = metrics.get('train_acc_epoch', float('nan'))
            val_loss = metrics.get('val_loss_epoch', float('nan'))
            val_acc = metrics.get('val_acc_epoch', float('nan'))

            self.train_history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
            self.val_history.append({'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}) # Keeping val_history as well if needed separately
            logger.info(f"Validation epoch {epoch} metrics recorded: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        except Exception as e:
            logger.error(f"Error in on_validation_epoch_end for model {self.model_name}, epoch {self.current_epoch}: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Exiting on_validation_epoch_end for epoch {self.current_epoch}, model: {self.model_name}")


    def configure_optimizers(self):
        logger.info(f"Configuring optimizers for model: {self.model_name} with learning rate: {self.learning_rate}")
        try:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            logger.info("AdamW optimizer configured.")
            return optimizer
        except Exception as e:
            logger.error(f"Error configuring optimizers for model {self.model_name}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.info(f"Exiting configure_optimizers for model: {self.model_name}")

    def compute_grad_norm(self):
        """Computes the gradient norm for logging."""
        norm_type = 2.0 # You can configure norm type if needed
        norms = [torch.linalg.norm(p.grad.detach(), norm_type) for p in self.parameters() if p.grad is not None]
        if len(norms) == 0:
            return torch.tensor(0.0)
        total_norm = torch.linalg.norm(torch.stack(norms), norm_type)
        return total_norm