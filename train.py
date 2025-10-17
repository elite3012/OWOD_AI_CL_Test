"""
Training Script for Enhanced OWOD with Continual Learning
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.defaults import get_cfg
from models.backbones.clip_backbone import build_clip_backbone
from models.continual_learning.memory_replay import build_memory_buffer
from models.continual_learning.knowledge_distillation import build_knowledge_distillation


def setup_logger(output_dir):
    """Setup logger for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Enhanced OWOD with Continual Learning')
    
    parser.add_argument(
        '--config-file',
        default='',
        metavar='FILE',
        help='path to config file',
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='number of GPUs to use',
    )
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='output directory for checkpoints and logs',
    )
    parser.add_argument(
        '--prev-model',
        default='',
        help='path to previous task model for continual learning',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume training from checkpoint',
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    return parser.parse_args()


def build_model(cfg):
    """
    Build the complete OWOD model with all components
    """
    logger = logging.getLogger(__name__)
    logger.info("Building model...")
    
    # Build backbone
    if cfg.FOUNDATION_MODEL.TYPE == "clip":
        backbone = build_clip_backbone(cfg)
        logger.info(f"Built CLIP backbone: {cfg.FOUNDATION_MODEL.CLIP.MODEL_NAME}")
    else:
        raise NotImplementedError(f"Backbone {cfg.FOUNDATION_MODEL.TYPE} not implemented")
    
    # TODO: Add ROI heads, detection heads, etc.
    # For now, return just the backbone
    
    return backbone


def build_optimizer(cfg, model):
    """Build optimizer"""
    if cfg.PEFT.ENABLED:
        # Only optimize PEFT parameters
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param)
        logger = logging.getLogger(__name__)
        logger.info(f"Training {len(params)} parameter groups with PEFT")
    else:
        params = model.parameters()
    
    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=0.9,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.SOLVER.OPTIMIZER}")
    
    return optimizer


def build_scheduler(cfg, optimizer):
    """Build learning rate scheduler"""
    if cfg.SOLVER.LR_SCHEDULER == "WarmupMultiStepLR":
        # TODO: Implement custom warmup scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA,
        )
    elif cfg.SOLVER.LR_SCHEDULER == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.COSINE_ANNEALING_T_MAX,
        )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    memory_buffer,
    distillation,
    teacher_model,
    cfg,
    epoch,
    device,
):
    """Train for one epoch"""
    model.train()
    logger = logging.getLogger(__name__)
    
    for iteration, batch in enumerate(data_loader):
        # Move data to device
        images = batch['images'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute task loss (placeholder - needs actual loss implementation)
        # task_loss = compute_detection_loss(outputs, targets)
        task_loss = torch.tensor(0.0, device=device)  # Placeholder
        
        # Add replay samples if available
        if memory_buffer and not memory_buffer.is_empty():
            replay_images, replay_targets = memory_buffer.sample(
                batch_size=int(len(images) * cfg.CONTINUAL_LEARNING.REPLAY.BATCH_RATIO),
                device=device,
            )
            if replay_images is not None:
                # Forward on replay samples
                replay_outputs = model(replay_images)
                # replay_loss = compute_detection_loss(replay_outputs, replay_targets)
                replay_loss = torch.tensor(0.0, device=device)  # Placeholder
                task_loss = task_loss + replay_loss
        
        # Knowledge distillation
        if distillation and teacher_model is not None:
            total_loss, loss_dict = distillation(
                model,
                teacher_model,
                images,
                targets,
                task_loss,
            )
        else:
            total_loss = task_loss
            loss_dict = {'total_loss': total_loss, 'task_loss': task_loss}
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE,
            )
        
        optimizer.step()
        
        # Logging
        if iteration % cfg.LOGGING.LOG_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch}] Iter [{iteration}] "
                f"Loss: {total_loss.item():.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        # Update memory buffer
        if memory_buffer and iteration % 100 == 0:
            with torch.no_grad():
                # Extract features for exemplar selection
                features = outputs  # Placeholder
                memory_buffer.update(images, targets, features)
    
    if scheduler:
        scheduler.step()


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(f"Command line args: {args}")
    
    # Load config
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    logger.info(f"Config:\n{cfg}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.num_gpus > 0 else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    model = build_model(cfg)
    model = model.to(device)
    
    # Load teacher model for continual learning
    teacher_model = None
    if args.prev_model and os.path.exists(args.prev_model):
        logger.info(f"Loading previous model from: {args.prev_model}")
        teacher_model = build_model(cfg)
        teacher_model.load_state_dict(torch.load(args.prev_model))
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    
    # Build continual learning components
    memory_buffer = None
    if cfg.CONTINUAL_LEARNING.REPLAY.ENABLED:
        memory_buffer = build_memory_buffer(cfg)
        logger.info(f"Built memory buffer with size: {cfg.CONTINUAL_LEARNING.REPLAY.MEMORY_SIZE}")
    
    distillation = None
    if cfg.CONTINUAL_LEARNING.DISTILLATION.ENABLED and teacher_model is not None:
        distillation = build_knowledge_distillation(cfg)
        logger.info(f"Built knowledge distillation with alpha: {cfg.CONTINUAL_LEARNING.DISTILLATION.ALPHA}")
    
    # TODO: Build data loaders
    # data_loader = build_data_loader(cfg)
    
    logger.info("Starting training...")
    
    # Training loop
    num_epochs = cfg.SOLVER.MAX_ITER // 1000  # Placeholder
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # TODO: Uncomment when data loader is ready
        # train_one_epoch(
        #     model,
        #     data_loader,
        #     optimizer,
        #     scheduler,
        #     memory_buffer,
        #     distillation,
        #     teacher_model,
        #     cfg,
        #     epoch,
        #     device,
        # )
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to: {checkpoint_path}")
            
            if memory_buffer:
                buffer_path = os.path.join(args.output_dir, f'memory_buffer_epoch_{epoch+1}.pth')
                memory_buffer.save(buffer_path)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training complete. Final model saved to: {final_model_path}")
    
    if memory_buffer:
        buffer_path = os.path.join(args.output_dir, 'memory_buffer_final.pth')
        memory_buffer.save(buffer_path)
        logger.info(f"Memory buffer saved to: {buffer_path}")


if __name__ == '__main__':
    main()
