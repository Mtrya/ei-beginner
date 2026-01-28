"""
Push-T Environment with Gymnasium API
A 2D planar pushing task using PyMunk physics simulator.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
from typing import Optional, Tuple, Dict, Any


def pymunk_to_shapely(body: pymunk.Body, shapes: list) -> sg.MultiPolygon:
    """Convert pymunk body and shapes to shapely geometry."""
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    return sg.MultiPolygon(geoms)


class PushTEnv(gym.Env):
    """
    Push-T environment where an agent pushes a T-shaped block to a goal region.
    
    Observation:
        - state: [agent_x, agent_y, block_x, block_y, block_angle] (5D)
        - image: RGB image of size (96, 96, 3) if render_mode='rgb_array'
    
    Action:
        - Continuous 2D position control [x, y] in pixel space [0, 512]
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        observation_type: str = "state",  # "state" or "image"
        render_size: int = 96,
        legacy: bool = False,
        block_cog: Optional[Vec2d] = None,
        damping: Optional[float] = None,
        render_action: bool = True,
    ):
        super().__init__()
        
        self.window_size = 512  # PyGame window size
        self.render_size = render_size
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.render_action = render_action
        
        # Physics parameters
        self.sim_hz = 100
        self.k_p, self.k_v = 100, 20  # PD control gains
        self.control_hz = self.metadata["render_fps"]
        
        # Optional parameters
        self.legacy = legacy
        self.block_cog = block_cog
        self.damping = damping
        
        # Define observation and action spaces
        if observation_type == "state":
            # [agent_x, agent_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([512, 512, 512, 512, 2 * np.pi], dtype=np.float32),
                shape=(5,),
                dtype=np.float32,
            )
        else:  # image
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                    low=0, high=255,
                    shape=(render_size, render_size, 3),
                    dtype=np.uint8
                ),
                "agent_pos": spaces.Box(
                    low=0, high=512,
                    shape=(2,),
                    dtype=np.float32
                )
            })
        
        # Action: target position for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([512, 512], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        
        # Rendering
        self.window = None
        self.clock = None
        self.screen = None
        
        # Environment state
        self.space = None
        self.agent = None
        self.block = None
        self.goal_pose = None
        self.goal_color = (153, 255, 153)  # Light green
        self.n_contact_points = 0
        self.success_threshold = 0.95
        self.latest_action = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Setup physics
        self._setup()
        
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # Random initial state
        state = np.array([
            self.np_random.integers(50, 450),
            self.np_random.integers(50, 450),
            self.np_random.integers(100, 400),
            self.np_random.integers(100, 400),
            self.np_random.uniform(-np.pi, np.pi),
        ], dtype=np.float32)
        
        self._set_state(state)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        
        if action is not None:
            self.latest_action = action.copy()
            action = action.astype(np.float32)
            
            for i in range(n_steps):
                # PD control to move agent toward target position
                action_vec = Vec2d(float(action[0]), float(action[1]))
                acceleration = (
                    self.k_p * (action_vec - self.agent.position) +
                    self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                )
                self.agent.velocity += acceleration * dt
                
                # Step physics
                self.space.step(dt)
        
        # Compute reward based on coverage
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        
        terminated = coverage > self.success_threshold
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        info["coverage"] = coverage
        
        return observation, float(reward), terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()
        return None
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def _setup(self):
        """Setup the physics simulation."""
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        
        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        
        # Add agent (blue circle)
        mass = 1
        radius = 15
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = 256, 256
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("dodgerblue")
        self.space.add(body, shape)
        self.agent = body
        
        # Add T-shaped block
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = 256, 256
        
        # T-shape: vertical bar + horizontal bar
        vertices_top = [(-25, 50), (25, 50), (25, 30), (-25, 30)]
        vertices_bar = [(-5, 30), (5, 30), (5, -50), (-5, -50)]
        
        shape_top = pymunk.Poly(body, vertices_top)
        shape_bar = pymunk.Poly(body, vertices_bar)
        shape_top.color = pygame.Color("lightcoral")
        shape_bar.color = pygame.Color("lightcoral")
        
        self.space.add(body, shape_top, shape_bar)
        self.block = body
        
        # Set goal pose
        self.goal_pose = np.array([256, 256, 0], dtype=np.float32)
        
        # Add collision handler
        handler = self.space.add_default_collision_handler()
        handler.post_solve = self._handle_collision
    
    def _add_segment(self, a: Tuple, b: Tuple, thickness: float):
        """Add a wall segment."""
        shape = pymunk.Segment(self.space.static_body, a, b, thickness)
        shape.color = pygame.Color("lightgray")
        self.space.add(shape)
        return shape
    
    def _handle_collision(self, arbiter, space, data):
        """Collision callback."""
        self.n_contact_points += len(arbiter.contact_point_set.points)
    
    def _set_state(self, state: np.ndarray):
        """Set environment state."""
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        
        self.agent.position = Vec2d(float(pos_agent[0]), float(pos_agent[1]))
        
        if self.legacy:
            self.block.position = Vec2d(float(pos_block[0]), float(pos_block[1]))
            self.block.angle = float(rot_block)
        else:
            self.block.angle = float(rot_block)
            self.block.position = Vec2d(float(pos_block[0]), float(pos_block[1]))
        
        # Let physics take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        if self.observation_type == "state":
            return np.array(
                list(self.agent.position) +
                list(self.block.position) +
                [self.block.angle % (2 * np.pi)],
                dtype=np.float32
            )
        else:  # image
            img = self._render_frame()
            return {
                "image": img,
                "agent_pos": np.array(self.agent.position, dtype=np.float32)
            }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        n_steps = self.sim_hz // self.control_hz
        n_contacts = int(np.ceil(self.n_contact_points / n_steps))
        
        return {
            "pos_agent": np.array(self.agent.position, dtype=np.float32),
            "vel_agent": np.array(self.agent.velocity, dtype=np.float32),
            "block_pose": np.array(
                list(self.block.position) + [self.block.angle],
                dtype=np.float32
            ),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contacts,
        }
    
    def _get_goal_pose_body(self, pose: np.ndarray) -> pymunk.Body:
        """Create a body representing the goal pose."""
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = Vec2d(float(pose[0]), float(pose[1]))
        body.angle = float(pose[2])
        return body
    
    def _render_frame(self) -> np.ndarray:
        """Render a single frame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Push-T")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        
        # Draw goal pose
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), canvas)
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)
        
        # Draw agent and block
        draw_options = pymunk.pygame_util.DrawOptions(canvas)
        self.space.debug_draw(draw_options)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
        
        # Convert to numpy array and resize
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        img = cv2.resize(img, (self.render_size, self.render_size))
        
        # Optionally draw action marker
        if self.render_action and self.latest_action is not None:
            action = self.latest_action.copy()
            coord = (action / 512 * self.render_size).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img, tuple(coord),
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness
            )
        
        return img
