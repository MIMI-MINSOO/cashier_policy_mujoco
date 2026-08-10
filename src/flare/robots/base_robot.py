from abc import ABC, abstractmethod
import numpy as np


class BaseRobot(ABC):
    """Base interface for robot control. All robots must implement these methods."""

    @abstractmethod
    def connect(self):
        """Connect to the robot hardware."""
        pass

    @abstractmethod
    def disconnect(self):
        """Safely disconnect from the robot."""
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Read current robot state (joint positions + gripper).

        Returns:
            np.ndarray: State vector. Shape depends on robot DOF.
        """
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray):
        """Send action command to the robot.

        Args:
            action: Action vector (joint positions + gripper).
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        pass

    @abstractmethod
    def move_to_eval_pose(self, duration: float = 5.0):
        """Move robot to task-ready evaluation pose (defined per-robot).

        Args:
            duration: Approximate time (seconds) to allow the move to complete.
        """
        pass

    @abstractmethod
    def reset_filter(self):
        """Reset any internal smoothing/interpolation state (e.g. between episodes)."""
        pass
