"""
Evaluation script for Diffusion Policy on Push-T task.
Runs the trained policy in the environment and visualizes performance.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from collections import deque

from src.envs.pusht import PushTEnv
from src.models.diffusion_policy import DiffusionPolicy


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = DiffusionPolicy(
        image_shape=(3, 96, 96),
        lowdim_dim=2,
        action_dim=2,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        num_train_timesteps=100,
        num_inference_timesteps=10,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set action stats
    if 'action_stats' in checkpoint:
        model.set_action_stats(checkpoint['action_stats'])
    
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model


def evaluate_episode(
    env: PushTEnv,
    model: DiffusionPolicy,
    device: torch.device,
    max_steps: int = 300,
    render: bool = True,
) -> dict:
    """Run one episode with the trained policy."""
    obs_dict, _ = env.reset()
    
    # Initialize observation buffer
    obs_buffer = deque(maxlen=model.n_obs_steps)
    
    # Fill buffer with initial observation
    for _ in range(model.n_obs_steps):
        obs_buffer.append(obs_dict)
    
    # Action buffer for receding horizon control
    action_buffer = None
    action_idx = 0
    
    total_reward = 0
    max_coverage = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Predict actions when buffer is empty
        if action_buffer is None or action_idx >= model.n_action_steps:
            # Prepare observation batch
            images = []
            agent_positions = []
            
            for obs in obs_buffer:
                img = obs['image']  # (96, 96, 3)
                # Normalize to [-1, 1]
                img = (img.astype(np.float32) - 127.5) / 127.5
                # Convert to (C, H, W)
                img = np.transpose(img, (2, 0, 1))
                images.append(img)
                
                agent_pos = obs['agent_pos']
                agent_positions.append(agent_pos)
            
            # Stack into tensors
            images = np.stack(images)  # (T_obs, C, H, W)
            agent_positions = np.stack(agent_positions)  # (T_obs, 2)
            
            # Add batch dimension
            images = torch.from_numpy(images).unsqueeze(0).float().to(device)
            agent_positions = torch.from_numpy(agent_positions).unsqueeze(0).float().to(device)
            
            # Predict actions
            with torch.no_grad():
                obs_input = {
                    'image': images,
                    'agent_pos': agent_positions,
                }
                action_pred = model.predict_action(obs_input)  # (1, horizon, 2)
            
            # Extract actions
            action_buffer = action_pred[0].cpu().numpy()  # (horizon, 2)
            action_idx = 0
        
        # Execute action
        action = action_buffer[action_idx]
        action_idx += 1
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update metrics
        total_reward += reward
        max_coverage = max(max_coverage, info['coverage'])
        
        # Add to observation buffer
        obs_buffer.append(obs_dict)
        
        # Render
        if render:
            env.render()
        
        step += 1
    
    return {
        'total_reward': total_reward,
        'max_coverage': max_coverage,
        'steps': step,
        'success': max_coverage > 0.95,  # Success threshold
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Create environment
    env = PushTEnv(
        observation_type='image',
        render_mode='human' if args.render else None,
        render_size=96,
    )
    
    print(f"\nRunning {args.num_episodes} episodes...")
    
    results = []
    for episode in range(args.num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{args.num_episodes}")
        print(f"{'='*50}")
        
        result = evaluate_episode(
            env,
            model,
            device,
            max_steps=args.max_steps,
            render=args.render,
        )
        
        results.append(result)
        
        print(f"Total reward: {result['total_reward']:.3f}")
        print(f"Max coverage: {result['max_coverage']:.3f}")
        print(f"Steps: {result['steps']}")
        print(f"Success: {result['success']}")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("Summary Statistics")
    print(f"{'='*50}")
    
    rewards = [r['total_reward'] for r in results]
    coverages = [r['max_coverage'] for r in results]
    successes = [r['success'] for r in results]
    
    print(f"Average reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Average max coverage: {np.mean(coverages):.3f} ± {np.std(coverages):.3f}")
    print(f"Success rate: {np.mean(successes):.1%}")
    
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/checkpoints/best_model.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    
    args = parser.parse_args()
    main(args)
