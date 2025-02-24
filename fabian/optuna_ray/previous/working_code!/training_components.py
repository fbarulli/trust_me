# training_components.py
import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import traceback
from typing import Dict, Optional
from torch.cuda.amp import GradScaler
from torch_ema import ExponentialMovingAverage


logger = logging.getLogger(__name__)

class TrainingComponents(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.validation_outputs = []
        self.use_ema = config.get('regularization', {}).get('use_ema', False)
        self.rdrop_loss = None  # This should be initialized in the child class
        self.gpu_tracker = None  # This should be initialized in the child class
        self.loss_scaler = GradScaler(
            enabled=config['training']['fp16_training'],
            init_scale=config['training']['grad_scaler']['init_scale'],
            growth_factor=config['training']['grad_scaler']['growth_factor'],
            backoff_factor=config['training']['grad_scaler']['backoff_factor'],
            growth_interval=config['training']['grad_scaler']['growth_interval']
        )
        
        # Initialize training parameters
        self.grad_clip_val = config['training']['gradient_clip_val']
        self.grad_clip_algorithm = config['training']['gradient_clip_algorithm']
        self.error_if_nonfinite = config['training']['error_if_nonfinite']

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model."""
        prepared_inputs = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                prepared_inputs[k] = v
                continue
            
            if k in ['input_ids', 'attention_mask']:
                prepared_inputs[k] = v.to(self.device, dtype=torch.long, non_blocking=True)
            else:
                prepared_inputs[k] = v.to(self.device, non_blocking=True)
        return prepared_inputs

    def _track_gradients(self) -> tuple[float, dict]:
        """Track gradient statistics for all parameters."""
        grad_norms = {}
        total_grad_norm = 0.0
        num_params_with_grad = 0
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms[name] = grad_norm
                    total_grad_norm += grad_norm ** 2
                    num_params_with_grad += 1
        
        if num_params_with_grad > 0:
            total_grad_norm = (total_grad_norm ** 0.5) / num_params_with_grad
            
        return total_grad_norm, grad_norms

    def _log_metrics(self, loss: torch.Tensor, accuracy: torch.Tensor, total_grad_norm: float):
        """Log training metrics."""
        self.log('train_loss', loss.item(), prog_bar=True, sync_dist=True)
        self.log('train_acc', accuracy.item(), prog_bar=True)
        self.log('grad_norm', total_grad_norm, prog_bar=True)
        
        if torch.cuda.is_available():
            self.log('gpu_memory_allocated', torch.cuda.memory_allocated() / 1024**2)
            self.log('gpu_memory_cached', torch.cuda.memory_reserved() / 1024**2)

    def _compute_loss(self, logits1: torch.Tensor, logits2: Optional[torch.Tensor], 
                     labels: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute the combined loss."""
        # Base cross entropy loss
        ce_loss = F.cross_entropy(logits1, labels)
        
        # Add RDrop loss if applicable
        if logits2 is not None and self.rdrop_loss is not None:
            rdrop_loss = self.rdrop_loss(logits1, logits2, labels)
            loss = ce_loss + self.config['regularization']['rdrop_alpha'] * rdrop_loss
        else:
            loss = ce_loss
        
        # Apply gradient accumulation scaling
        return loss / self.config['training']['gradient_accumulation_steps']

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        batch = self._prepare_inputs(batch)
        labels = batch.pop('labels')

        # Track initial memory state
        if self.gpu_tracker:
            self.gpu_tracker.track_step("pre_forward", model=self)
        
        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.cuda.amp.autocast(enabled=self.config['training']['fp16_training']):
                logits1 = self(**batch)
                
                # Apply RDrop if configured
                if self.rdrop_loss and self.current_epoch >= self.config['training']['rdrop_start_epoch']:
                    with torch.no_grad():
                        logits2 = self(**batch)
                    loss = self._compute_loss(logits1, logits2, labels, batch_idx)
                else:
                    loss = self._compute_loss(logits1, None, labels, batch_idx)

            # Scale loss and compute gradients
            scaled_loss = self.loss_scaler.scale(loss)
            self.manual_backward(scaled_loss)
            
            # Track gradients after backward pass
            total_grad_norm, grad_norms = self._track_gradients()
            
            # Track memory after backward pass
            if self.gpu_tracker:
                self.gpu_tracker.track_step("post_backward", model=self, 
                                          local_vars={'loss': loss, 'logits': logits1})

            # Optimizer step with gradient clipping
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # Unscale gradients
                self.loss_scaler.unscale_(optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=self.grad_clip_val,
                    norm_type=2.0 if self.grad_clip_algorithm == 'norm' else float('inf'),
                    error_if_nonfinite=self.error_if_nonfinite
                )
                
                # Step optimizer and update scaler
                scale = self.loss_scaler.get_scale()
                self.loss_scaler.step(optimizer)
                self.loss_scaler.update()
                
                # Step scheduler if optimization was successful
                if self.loss_scaler.get_scale() == scale:
                    scheduler.step()

            # Calculate accuracy and log metrics
            accuracy = (logits1.argmax(dim=-1) == labels).float().mean()
            self._log_metrics(loss, accuracy, total_grad_norm)

            return loss
            
        except RuntimeError as e:
            logger.error(f"Runtime error during training step: {str(e)}")
            self.log('training_error', 1.0, prog_bar=True)
            raise e
        finally:
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Execute validation step with memory and gradient handling."""
        try:
            batch = self._prepare_inputs(batch)
            
            with torch.cuda.amp.autocast(enabled=self.config['training']['fp16_training']):
                labels = batch.pop('labels')
                logits = self(**batch)
                loss = F.cross_entropy(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = torch.tensor(accuracy_score(labels.cpu(), preds.cpu()))

            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
            self.log('val_acc', acc, prog_bar=True, sync_dist=True)
            
            self.validation_outputs.append({
                'val_loss': loss.detach(),
                'val_acc': acc,
                'logits_mean': logits.mean().item(),
                'logits_std': logits.std().item()
            })
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during validation step: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_start(self):
        """Called at the start of validation epoch."""
        self.validation_outputs = []
        if self.use_ema and hasattr(self, 'ema'):  # Check if ema exists
            self.ema.store()
            self.ema.copy_to()


    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        try:
            if not self.validation_outputs:
                logger.warning("No validation outputs collected")
                avg_loss = torch.tensor(float('inf'))
                avg_acc = torch.tensor(0.0)
            else:
                avg_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
                avg_acc = torch.stack([x['val_acc'] for x in self.validation_outputs]).mean()

            self.log('avg_val_loss', avg_loss, prog_bar=True, sync_dist=True)
            self.log('avg_val_acc', avg_acc, prog_bar=True, sync_dist=True)
            self.validation_outputs = []

            if self.use_ema and hasattr(self, 'ema'):  # Check if ema exists
                self.ema.restore()

        except Exception as e:
            logger.error(f"Error during validation epoch end: {str(e)}")
            logger.error(traceback.format_exc())
            self.log('avg_val_loss', torch.tensor(float('inf')), prog_bar=True, sync_dist=True)
            self.log('avg_val_acc', torch.tensor(0.0), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Extract relevant config sections
        opt_config = self.config['network_training']['optimizer']
        lr_config = self.config['network_training']['lr_scheduler']
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=float(opt_config['learning_rate']),
            weight_decay=float(self.config['regularization']['weight_decay']),
            betas=(float(opt_config['beta1']), float(opt_config['beta2']))
        )
        
        # Configure learning rate scheduler
        scheduler = self._configure_scheduler(optimizer, lr_config)
        
        # Return based on scheduler type
        if lr_config['type'] == 'onecycle':
            return [optimizer], [{
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'train_loss'
            }]
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def _configure_scheduler(self, optimizer, lr_config):
        """Configure different scheduler types"""
        if lr_config['type'] == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.config['search_space']['max_lr']['max']),
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=float(lr_config['pct_start']),
                div_factor=float(lr_config['div_factor']),
                final_div_factor=float(lr_config['final_div_factor']),
                anneal_strategy=lr_config['anneal_strategy']
            )
            
        if lr_config['type'] == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(lr_config['factor']),
                patience=int(lr_config['patience']),
                verbose=lr_config['verbose']
            )
            
        # Default no-op scheduler
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

    def on_train_start(self):
        """Called when training begins."""
        if self.use_ema and not hasattr(self, 'ema'):
            try:
                from torch_ema import ExponentialMovingAverage
                logger.info("Initializing EMA")
                self.ema = ExponentialMovingAverage(
                    self.parameters(),
                    decay=self.config['regularization']['ema_decay']
                )
                logger.info("EMA initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EMA: {str(e)}")
                self.use_ema = False  # Disable EMA if initialization fails

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called at the end of training batch."""
        if self.use_ema and hasattr(self, 'ema'):  # Check if ema exists
            self.ema.update()

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.use_ema and hasattr(self, 'ema'):  # Check if ema exists
            self.ema.apply_shadow()
            # ... any other epoch end code ...
            self.ema.restore()
