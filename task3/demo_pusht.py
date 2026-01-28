"""
Interactive demo for Push-T environment.

Controls:
- Move mouse to control the agent (blue circle)
- Mouse proximity activates control
- Press 'Q' to quit
- Press 'R' to reset

The agent (blue circle) should push the T-shaped block (red) into the goal region (green).
"""

import sys
sys.path.insert(0, 'task3/src')

import numpy as np
import pygame
from envs.pusht import PushTEnv


def main():
    """Run interactive Push-T demo."""
    print("=" * 60)
    print("Push-T Environment Demo")
    print("=" * 60)
    print("\nControls:")
    print("  - Move mouse to control the agent (blue circle)")
    print("  - Mouse must be close to agent to activate control")
    print("  - Press 'Q' to quit")
    print("  - Press 'R' to reset")
    print("\nGoal: Push the T-block (red) into the green region")
    print("=" * 60 + "\n")
    
    # Create environment with human rendering
    env = PushTEnv(
        render_mode="human",
        observation_type="state",
        render_size=512,
        render_action=True,
    )
    
    # Initialize pygame for mouse input
    pygame.init()
    clock = pygame.time.Clock()
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial state: agent={info['pos_agent']}, block={info['block_pose'][:2]}")
    
    # Control parameters
    control_active = False
    control_radius = 50  # Pixels within which control activates
    
    episode_reward = 0
    step_count = 0
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    episode_reward = 0
                    step_count = 0
                    control_active = False
                    print("\nEnvironment reset!")
                    print(f"Initial state: agent={info['pos_agent']}, block={info['block_pose'][:2]}")
        
        # Get mouse position (in pygame coordinates)
        mouse_pos = pygame.mouse.get_pos()
        
        # Convert to environment coordinates (flip Y axis)
        mouse_x = mouse_pos[0]
        mouse_y = mouse_pos[1]
        
        # Check if mouse is close to agent to activate control
        agent_pos = info['pos_agent']
        distance = np.sqrt((mouse_x - agent_pos[0])**2 + (mouse_y - agent_pos[1])**2)
        
        if distance < control_radius or control_active:
            control_active = True
            action = np.array([mouse_x, mouse_y], dtype=np.float32)
        else:
            # No action (stay in place)
            action = agent_pos
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Display info every 10 steps
        if step_count % 10 == 0:
            coverage = info['coverage']
            print(f"Step {step_count}: reward={episode_reward:.3f}, coverage={coverage:.3f}, contacts={info['n_contacts']}")
        
        # Check if episode ended
        if terminated:
            print(f"\n{'='*60}")
            print(f"SUCCESS! Task completed in {step_count} steps!")
            print(f"Total reward: {episode_reward:.3f}")
            print(f"Coverage: {info['coverage']:.3f}")
            print(f"{'='*60}\n")
            
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            control_active = False
        
        # Render
        env.render()
        
        # Control frame rate
        clock.tick(10)  # 10 FPS
    
    env.close()
    pygame.quit()
    print("\nDemo ended. Thanks for playing!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        pygame.quit()
