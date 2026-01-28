"""Dataset loader for Push-T demonstrations stored in Zarr format."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
import zarr
from pathlib import Path


class PushTImageDataset(Dataset):
    """
    Dataset for Push-T with image observations.
    
    Data format in Zarr:
        data/img: (N, 96, 96, 3) uint8 images
        data/state: (N, 5) float32 states [agent_x, agent_y, block_x, block_y, block_angle]
        data/action: (N, 2) float32 actions [target_x, target_y]
        meta/episode_ends: (num_episodes,) int64 episode end indices
    
    Returns sequences of length `horizon` with padding.
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        is_val: bool = False,
    ):
        """
        Args:
            zarr_path: Path to Zarr dataset
            horizon: Total prediction horizon
            n_obs_steps: Number of observation steps
            n_action_steps: Number of action steps to execute
            seed: Random seed for train/val split
            val_ratio: Validation split ratio
            max_train_episodes: Maximum number of training episodes
            is_val: Whether this is validation dataset
        """
        super().__init__()
        
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.pad_before = n_obs_steps - 1
        self.pad_after = n_action_steps - 1
        
        # Load dataset
        zarr_path = Path(zarr_path)
        if zarr_path.suffix == '.zip':
            store = zarr.ZipStore(zarr_path, mode='r')
        else:
            store = str(zarr_path)
        
        root = zarr.open(store, mode='r')
        
        # Load data
        self.images = root['data']['img'][:]  # (N, 96, 96, 3)
        self.states = root['data']['state'][:]  # (N, 5)
        self.actions = root['data']['action'][:]  # (N, 2)
        self.episode_ends = root['meta']['episode_ends'][:]  # (num_episodes,)
        
        # Compute episode starts
        self.episode_starts = np.zeros(len(self.episode_ends), dtype=np.int64)
        self.episode_starts[1:] = self.episode_ends[:-1]
        
        # Train/val split
        rng = np.random.RandomState(seed)
        n_episodes = len(self.episode_ends)
        val_mask = np.zeros(n_episodes, dtype=bool)
        val_indices = rng.choice(
            n_episodes,
            size=int(n_episodes * val_ratio),
            replace=False
        )
        val_mask[val_indices] = True
        
        if is_val:
            self.episode_mask = val_mask
        else:
            train_mask = ~val_mask
            if max_train_episodes is not None and max_train_episodes < train_mask.sum():
                # Downsample training episodes
                train_indices = np.where(train_mask)[0]
                selected = rng.choice(
                    train_indices,
                    size=max_train_episodes,
                    replace=False
                )
                train_mask = np.zeros(n_episodes, dtype=bool)
                train_mask[selected] = True
            self.episode_mask = train_mask
        
        # Compute valid indices for sampling
        self.indices = []
        for episode_idx in np.where(self.episode_mask)[0]:
            start_idx = self.episode_starts[episode_idx]
            end_idx = self.episode_ends[episode_idx]
            episode_length = end_idx - start_idx
            
            # Each episode can generate multiple samples
            for i in range(episode_length):
                self.indices.append((episode_idx, start_idx + i))
        
        print(f"Loaded dataset: {len(self.indices)} samples from "
              f"{self.episode_mask.sum()} episodes")
        
        # Compute normalization statistics
        self._compute_normalization()
    
    def _compute_normalization(self):
        """Compute normalization statistics for actions and agent positions."""
        # Action normalization (use all data for consistency)
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0)
        self.action_std = np.where(self.action_std < 1e-6, 1.0, self.action_std)
        
        # Agent position normalization (first 2 elements of state)
        agent_pos = self.states[:, :2]
        self.agent_pos_mean = agent_pos.mean(axis=0)
        self.agent_pos_std = agent_pos.std(axis=0)
        self.agent_pos_std = np.where(self.agent_pos_std < 1e-6, 1.0, self.agent_pos_std)
        
        # Image normalization: scale to [-1, 1]
        self.image_mean = 127.5
        self.image_std = 127.5
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to zero mean and unit variance."""
        return (action - self.action_mean) / self.action_std
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action back to original scale."""
        return action * self.action_std + self.action_mean
    
    def normalize_agent_pos(self, agent_pos: np.ndarray) -> np.ndarray:
        """Normalize agent position."""
        return (agent_pos - self.agent_pos_mean) / self.agent_pos_std
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [-1, 1]."""
        return (image - self.image_mean) / self.image_std
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with sequences of observations and actions.
        
        Returns:
            Dictionary with keys:
                - 'image': (T_obs, 3, 96, 96) float32, normalized to [-1, 1]
                - 'agent_pos': (T_obs, 2) float32, normalized
                - 'action': (T_pred, 2) float32, normalized
        """
        episode_idx, frame_idx = self.indices[idx]
        
        start_idx = self.episode_starts[episode_idx]
        end_idx = self.episode_ends[episode_idx]
        
        # Sample observation sequence [frame_idx - pad_before, frame_idx + 1)
        obs_start = frame_idx - self.pad_before
        obs_end = frame_idx + 1
        
        # Sample action sequence [frame_idx, frame_idx + horizon)
        act_start = frame_idx
        act_end = frame_idx + self.horizon
        
        # Handle padding for observations
        obs_indices = np.arange(obs_start, obs_end)
        # Clip to episode boundaries and pad with first/last frame
        obs_indices = np.clip(obs_indices, start_idx, end_idx - 1)
        
        # Handle padding for actions
        act_indices = np.arange(act_start, act_end)
        # Pad with last action if exceeding episode
        act_indices = np.clip(act_indices, start_idx, end_idx - 1)
        
        # Get data
        images = self.images[obs_indices]  # (T_obs, 96, 96, 3)
        agent_pos = self.states[obs_indices, :2]  # (T_obs, 2)
        actions = self.actions[act_indices]  # (T_pred, 2)
        
        # Normalize
        images = self.normalize_image(images.astype(np.float32))
        agent_pos = self.normalize_agent_pos(agent_pos)
        actions = self.normalize_action(actions)
        
        # Convert to torch tensors
        # Images: (T, H, W, C) -> (T, C, H, W)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        agent_pos = torch.from_numpy(agent_pos).float()
        actions = torch.from_numpy(actions).float()
        
        return {
            'image': images,
            'agent_pos': agent_pos,
            'action': actions,
        }
    
    def get_val_dataset(self) -> 'PushTImageDataset':
        """Create validation dataset with same parameters."""
        return PushTImageDataset(
            zarr_path=self.zarr_path,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            seed=self.seed,
            val_ratio=self.val_ratio,
            max_train_episodes=None,
            is_val=True,
        )


class PushTStateDataset(Dataset):
    """
    Dataset for Push-T with low-dimensional state observations.
    Simpler version without images.
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        is_val: bool = False,
    ):
        super().__init__()
        
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
        # Load dataset
        zarr_path = Path(zarr_path)
        if zarr_path.suffix == '.zip':
            store = zarr.ZipStore(zarr_path, mode='r')
        else:
            store = str(zarr_path)
        
        root = zarr.open(store, mode='r')
        
        self.states = root['data']['state'][:]  # (N, 5)
        self.actions = root['data']['action'][:]  # (N, 2)
        self.episode_ends = root['meta']['episode_ends'][:]
        
        # Rest is similar to image dataset
        self.episode_starts = np.zeros(len(self.episode_ends), dtype=np.int64)
        self.episode_starts[1:] = self.episode_ends[:-1]
        
        # Train/val split
        rng = np.random.RandomState(seed)
        n_episodes = len(self.episode_ends)
        val_mask = np.zeros(n_episodes, dtype=bool)
        val_indices = rng.choice(
            n_episodes,
            size=int(n_episodes * val_ratio),
            replace=False
        )
        val_mask[val_indices] = True
        
        self.episode_mask = val_mask if is_val else ~val_mask
        
        # Compute indices
        self.indices = []
        for episode_idx in np.where(self.episode_mask)[0]:
            start_idx = self.episode_starts[episode_idx]
            end_idx = self.episode_ends[episode_idx]
            episode_length = end_idx - start_idx
            
            for i in range(episode_length):
                self.indices.append((episode_idx, start_idx + i))
        
        # Normalization
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)
        
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0)
        self.action_std = np.where(self.action_std < 1e-6, 1.0, self.action_std)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, frame_idx = self.indices[idx]
        
        start_idx = self.episode_starts[episode_idx]
        end_idx = self.episode_ends[episode_idx]
        
        # Sample sequences
        obs_start = frame_idx - (self.n_obs_steps - 1)
        obs_end = frame_idx + 1
        act_start = frame_idx
        act_end = frame_idx + self.horizon
        
        obs_indices = np.clip(np.arange(obs_start, obs_end), start_idx, end_idx - 1)
        act_indices = np.clip(np.arange(act_start, act_end), start_idx, end_idx - 1)
        
        states = self.states[obs_indices]
        actions = self.actions[act_indices]
        
        # Normalize
        states = (states - self.state_mean) / self.state_std
        actions = (actions - self.action_mean) / self.action_std
        
        return {
            'state': torch.from_numpy(states).float(),
            'action': torch.from_numpy(actions).float(),
        }
