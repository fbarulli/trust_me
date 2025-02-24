import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple, List
from transformers import AutoModel, AutoConfig
from torch.cuda.amp import GradScaler
from training_components import TrainingComponents
from torchmetrics.classification import Accuracy


logger = logging.getLogger(__name__)

class OptimizedModel(TrainingComponents):
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False  # Add this line

        
        # Load base model configuration
        self.model_config = AutoConfig.from_pretrained(
            config['model']['base_model_name'],
            num_labels=config['model']['num_labels']
        )
        
        # Initialize the base transformer model
        self.base_model = AutoModel.from_pretrained(
            config['model']['base_model_name'],
            config=self.model_config
        )
        
        # Get hidden size from config
        self.hidden_size = self.model_config.hidden_size
        
        # Initialize classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config['model_dropout_rate']),
            nn.Linear(self.hidden_size, config['model']['num_labels'])
        )
        
        # Initialize loss functions and metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=config['model']['num_labels'])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=config['model']['num_labels'])
        
        # Initialize loss scaler for mixed precision training
        self.loss_scaler = GradScaler(
            enabled=config['training']['fp16_training'],
            init_scale=config['training']['grad_scaler']['init_scale'],
            growth_factor=config['training']['grad_scaler']['growth_factor'],
            backoff_factor=config['training']['grad_scaler']['backoff_factor'],
            growth_interval=config['training']['grad_scaler']['growth_interval']
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        """Forward pass of the model."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Pass through classification head
        logits = self.classifier(pooled_output)
        
        return logits

    def _compute_loss(self, logits1: torch.Tensor, logits2: Optional[torch.Tensor], 
                     labels: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute the combined loss."""
        # Temperature scaling for logits1
        scaled_logits1 = logits1 / 1.5

        # Base cross entropy loss with temperature scaling
        ce_loss = F.cross_entropy(scaled_logits1, labels)
        
        # Add RDrop loss if applicable
        if logits2 is not None and self.rdrop_loss is not None:
            scaled_logits2 = logits2 / 1.5  # Temperature scaling for logits2
            rdrop_loss = self.rdrop_loss(scaled_logits1, scaled_logits2, labels)
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
        if hasattr(self, 'gpu_tracker') and self.gpu_tracker:
            self.gpu_tracker.track_step("pre_forward", model=self)
        
        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast('cuda', enabled=self.config['training']['fp16_training']):
                logits1 = self(**batch)
                
                # Apply RDrop if configured and we're past the start epoch
                use_rdrop = (
                    hasattr(self, 'rdrop_loss') and 
                    self.rdrop_loss is not None and 
                    self.current_epoch >= self.config['training'].get('rdrop_start_epoch', 0)
                )
                
                if use_rdrop:
                    with torch.no_grad():
                        logits2 = self(**batch)
                    loss = self._compute_loss(logits1, logits2, labels, batch_idx)
                else:
                    loss = self._compute_loss(logits1, None, labels, batch_idx)

                # Scale loss and compute gradients
                scaled_loss = self.loss_scaler.scale(loss) if hasattr(self, 'loss_scaler') else loss
                self.manual_backward(scaled_loss)
                
                # Track gradients after backward pass
                total_grad_norm, grad_norms = self._track_gradients()
                
                # Track memory after backward pass
                if hasattr(self, 'gpu_tracker') and self.gpu_tracker:
                    self.gpu_tracker.track_step("post_backward", model=self, 
                                              local_vars={'loss': loss, 'logits': logits1})

                # Optimizer step with gradient clipping
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    if hasattr(self, 'loss_scaler'):
                        # Unscale gradients
                        self.loss_scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=self.config['training']['gradient_clip_val'],
                        norm_type=2.0 if self.config['training']['gradient_clip_algorithm'] == 'norm' else float('inf'),
                        error_if_nonfinite=False
                    )
                    
                    # Step optimizer and update scaler
                    if hasattr(self, 'loss_scaler'):
                        scale = self.loss_scaler.get_scale()
                        self.loss_scaler.step(optimizer)
                        self.loss_scaler.update()
                        
                        # Step scheduler if optimization was successful
                        if self.loss_scaler.get_scale() == scale:
                            scheduler.step()
                    else:
                        optimizer.step()
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
    def _log_metrics(self, loss: torch.Tensor, accuracy: torch.Tensor, total_grad_norm: float):
        """Log comprehensive training metrics and hyperparameters."""
        # Training metrics
        self.log('train/loss', loss.item(), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log('train/accuracy', accuracy.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/grad_norm', total_grad_norm, prog_bar=True, on_step=True, on_epoch=True)
        
        # Learning rates and optimizer states
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True, on_epoch=True)
        
        # Network training parameters
        self.log('network_training/learning_rate', float(self.config['network_training']['optimizer']['learning_rate']))
        self.log('network_training/weight_decay', float(self.config['network_training']['optimizer']['weight_decay']))
        self.log('network_training/beta1', float(self.config['network_training']['optimizer']['beta1']))
        self.log('network_training/beta2', float(self.config['network_training']['optimizer']['beta2']))
        self.log('network_training/eps', float(self.config['network_training']['optimizer']['eps']))

        # Regularization parameters
        self.log('regularization/attention_dropout', float(self.config['regularization']['attention_dropout']))
        self.log('regularization/classifier_dropout', float(self.config['regularization']['classifier_dropout']))
        self.log('regularization/hidden_dropout', float(self.config['regularization']['hidden_dropout']))
        self.log('regularization/label_smoothing', float(self.config['regularization']['label_smoothing']))
        self.log('regularization/weight_decay', float(self.config['regularization']['weight_decay']))
        self.log('regularization/rdrop_alpha', float(self.config['regularization']['rdrop_alpha']))
        self.log('regularization/mixup_alpha', float(self.config['regularization']['mixup_alpha']))

        # Model architecture parameters
        self.log('model_architecture/unfrozen_layers', float(self.config['model_architecture']['unfrozen_layers']))
        self.log('model_architecture/pooling_dropout', float(self.config['model_architecture']['pooling_head']['dropout_prob']))

        # Training parameters
        self.log('training/gradient_accumulation_steps', float(self.config['training']['gradient_accumulation_steps']))
        self.log('training/gradient_clip_val', float(self.config['training']['gradient_clip_val']))
        self.log('training/initial_learning_rate', float(self.config['training']['initial_learning_rate']))
        
        # GPU metrics
        if torch.cuda.is_available():
            self.log('gpu/memory_allocated', torch.cuda.memory_allocated() / 1024**2, on_step=True)
            self.log('gpu/memory_reserved', torch.cuda.memory_reserved() / 1024**2, on_step=True)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Execute validation step with comprehensive logging."""
        try:
            batch = self._prepare_inputs(batch)
            
            with torch.amp.autocast('cuda', enabled=self.config['training']['fp16_training']):
                labels = batch.pop('labels')
                logits = self(**batch)
                loss = F.cross_entropy(logits, labels, 
                                    label_smoothing=float(self.config['regularization']['label_smoothing']))
                preds = torch.argmax(logits, dim=-1)
                acc = self.val_accuracy(preds, labels)

            # Basic validation metrics
            self.log('val/loss', float(loss), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            self.log('val/accuracy', float(acc), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            
            # Detailed logits statistics
            self.log('val/logits_mean', float(logits.mean()), on_epoch=True)
            self.log('val/logits_std', float(logits.std()), on_epoch=True)
            self.log('val/logits_max', float(logits.max()), on_epoch=True)
            self.log('val/logits_min', float(logits.min()), on_epoch=True)
            
            # Class distribution
            for i in range(self.config['model']['num_labels']):
                self.log(f'val/class_{i}_count', float((preds == i).sum()), on_epoch=True)
            
            # Confidence metrics
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1)[0]
            self.log('val/mean_confidence', float(confidence.mean()), on_epoch=True)
            
            # Model state metrics
            if float(self.config['training']['gradient_clip_val']) > 0:
                params_norm = torch.norm(torch.cat([p.view(-1) for p in self.parameters()]))
                self.log('val/params_norm', float(params_norm), on_epoch=True)
            
            # GPU metrics during validation
            if torch.cuda.is_available():
                self.log('val/gpu_memory_allocated', float(torch.cuda.memory_allocated() / 1024**2), on_step=True)
                self.log('val/gpu_memory_reserved', float(torch.cuda.memory_reserved() / 1024**2), on_step=True)
            
            return {
                'val_loss': float(loss),
                'val_acc': float(acc),
                'val_logits_mean': float(logits.mean()),
                'val_logits_std': float(logits.std()),
                'val_confidence_mean': float(confidence.mean())
            }
            
        except Exception as e:
            logger.error(f"Error during validation step: {str(e)}")
            logger.error(traceback.format_exc())
            return None









    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Create parameter groups with different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                        if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['regularization']['weight_decay']
            },
            {
                'params': [p for n, p in self.named_parameters() 
                        if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=float(self.config['regularization']['weight_decay']),
            betas=(float(self.config['network_training']['optimizer']['beta1']), 
                float(self.config['network_training']['optimizer']['beta2']))
        )
        
        # Initialize scheduler using the new method
        scheduler = self._configure_scheduler(optimizer, self.config['network_training']['lr_scheduler'])
        
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': None,
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}


    def get_progress_bar_dict(self):
        """Customize items to be displayed in the progress bar."""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
