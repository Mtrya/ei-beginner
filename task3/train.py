"""
Training script for Diffusion Policy on Push-T task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm
import argparse

from src.dataset import PushTImageDataset
from src.models.diffusion_policy import DiffusionPolicy


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        obs = {
            'image': batch['image'].to(device),
            'agent_pos': batch['agent_pos'].to(device),
        }
        action = batch['action'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.compute_loss(obs, action)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    for batch in dataloader:
        obs = {
            'image': batch['image'].to(device),
            'agent_pos': batch['agent_pos'].to(device),
        }
        action = batch['action'].to(device)
        
        loss = model.compute_loss(obs, action)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = PushTImageDataset(
        zarr_path=args.data_path,
        horizon=args.prediction_horizon,
        n_obs_steps=args.obs_horizon,
        n_action_steps=args.action_horizon,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_train_episodes=args.max_train_episodes,
        is_val=False,
    )
    
    val_dataset = PushTImageDataset(
        zarr_path=args.data_path,
        horizon=args.prediction_horizon,
        n_obs_steps=args.obs_horizon,
        n_action_steps=args.action_horizon,
        seed=args.seed,
        val_ratio=args.val_ratio,
        is_val=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = DiffusionPolicy(
        image_shape=(3, 96, 96),
        lowdim_dim=2,
        action_dim=2,
        horizon=args.prediction_horizon,
        n_obs_steps=args.obs_horizon,
        n_action_steps=args.action_horizon,
        num_train_timesteps=args.num_diffusion_iters,
        num_inference_timesteps=args.num_inference_steps,
    ).to(device)
    
    # Set action normalization stats
    action_stats = {
        'mean': torch.from_numpy(train_dataset.action_mean).to(device),
        'std': torch.from_numpy(train_dataset.action_std).to(device),
    }
    model.set_action_stats(action_stats)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr / 10,
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'action_stats': action_stats,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'action_stats': action_stats,
            }, best_path)
            print(f"New best model! Val loss: {val_loss:.4f}")
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'action_stats': action_stats,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default='data/pusht/pusht_cchi_v7_replay.zarr',
                       help='Path to Zarr dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model
    parser.add_argument('--obs_horizon', type=int, default=2,
                       help='Number of observation frames')
    parser.add_argument('--action_horizon', type=int, default=8,
                       help='Number of actions to execute')
    parser.add_argument('--prediction_horizon', type=int, default=16,
                       help='Total prediction horizon')
    parser.add_argument('--num_diffusion_iters', type=int, default=100,
                       help='Number of diffusion iterations for training')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                       help='Number of inference steps (DDIM)')
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_ratio', type=float, default=0.02,
                       help='Validation split ratio')
    parser.add_argument('--max_train_episodes', type=int, default=None,
                       help='Maximum training episodes (None for all)')
    
    args = parser.parse_args()
    main(args)
