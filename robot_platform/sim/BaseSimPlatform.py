"""Abstract base for MuJoCo simulation platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSimPlatform(ABC):
    """Interface for applying control and reading state from a MuJoCo sim."""

    def __init__(self, *args, **kwargs):
        self.is_valid = False

    def __str__(self) -> str:
        return f"Simulation Platform: {self.__class__.__name__}"

    def add_friction_and_inertial_correction_to_sim(self, *args, **kwargs):
        """Optionally inject friction / rotor-inertia correction torques.

        Default is a no-op. Subclasses may override.
        """
        return 0

    @abstractmethod
    def apply_sim_control(self, ctrl_data, mj_data):
        """Apply ``ctrl_data`` to the MuJoCo actuators in ``mj_data``."""

    @abstractmethod
    def sync_intr_data_from_sim(self, mj_data, intr_pub_que_dict: dict):
        """Publish platform-intrinsic measurements from the simulation."""

    @abstractmethod
    def reset(self, mj_data=None):
        """Reset the platform to its initial state."""

    @abstractmethod
    def get_col_info_data(self):
        """Return collision geometry descriptors for this platform."""
