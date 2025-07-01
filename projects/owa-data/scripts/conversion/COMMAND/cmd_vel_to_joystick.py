#!/usr/bin/env python3
"""
Convert CMD_VEL (command velocity) back to joystick input format.

This is useful for understanding the original human control patterns when the robot
was controlled via joystick during data collection.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class IsaacHoundbotJoystickConfig:
    """Configuration for Isaac_Houndbot joystick conversion based on actual teleop implementation."""
    # Isaac_Houndbot parameters from issac_sim_teleop_node.py
    max_forward_lin_vel: float = 2.5      # m/s
    max_backward_lin_vel: float = 0.625   # m/s (2.5 * 0.25)
    max_ang_vel: float = 1.4              # rad/s
    max_ang_correction: float = 1.8 / 1.4 # Angular correction factor when running
    backward_vel_reduction_rate: float = 0.5

    # Standard gamepad layout (simplified to 4 axes)
    # [left_stick_x, left_stick_y, right_stick_x, right_stick_y]
    linear_axis: int = 3   # Right stick Y-axis (forward/backward) - index 3 in 4-axis layout
    angular_axis: int = 0  # Left stick X-axis (left/right turn) - index 0 in 4-axis layout

    # Joystick axis ranges (gamepad: -1.0 to 1.0)
    axis_min: float = -1.0
    axis_max: float = 1.0

    # Deadzone for joystick (small for precise control)
    deadzone: float = 0.01

    # Status tracking (matches Isaac_Houndbot.Status)
    # STOP = 0, ROTATION = 1, RUN = 2


class IsaacHoundbotCmdVelToJoystick:
    """
    Convert CMD_VEL commands back to joystick input format for Isaac_Houndbot.

    Based on the actual implementation in issac_sim_teleop_node.py
    """

    def __init__(self, config: Optional[IsaacHoundbotJoystickConfig] = None):
        self.config = config or IsaacHoundbotJoystickConfig()
        # Track robot status for proper conversion
        self.current_status = "STOP"  # STOP, ROTATION, RUN

    def convert_single(self, linear_x: float, angular_z: float) -> Dict[str, float]:
        """
        Convert a single CMD_VEL command to joystick axes using Isaac_Houndbot logic.

        Args:
            linear_x: Linear velocity in m/s
            angular_z: Angular velocity in rad/s

        Returns:
            Dictionary with joystick axis values
        """
        # Determine robot status based on velocities
        self._update_status(linear_x, angular_z)

        # Convert based on Isaac_Houndbot's compute_cmd_vel logic (reverse)
        linear_rate, angular_rate = self._cmd_vel_to_rates(linear_x, angular_z)

        # Apply deadzone
        linear_rate = self._apply_deadzone(linear_rate)
        angular_rate = self._apply_deadzone(angular_rate)

        # Create simplified joystick message format (just 4 axes for main sticks)
        # [left_stick_x, left_stick_y, right_stick_x, right_stick_y]
        axes = [0.0] * 4

        # Map to standard gamepad layout (regardless of which axis is used in teleop)
        axes[0] = angular_rate  # Left stick X (for turning)
        axes[3] = linear_rate   # Right stick Y (for forward/backward)

        return {
            'axes': axes,  # Just 4 axes for the main sticks
            # No buttons since they're not used
            'linear_axis_value': linear_rate,
            'angular_axis_value': angular_rate,
            'original_linear_x': linear_x,
            'original_angular_z': angular_z,
            'robot_status': self.current_status
        }

    def _update_status(self, linear_x: float, angular_z: float):
        """Update robot status based on velocities (matches Isaac_Houndbot logic)."""
        err = 0.01
        is_stop = abs(linear_x) < err and abs(angular_z) < err

        if self.current_status == "STOP":
            if linear_x != 0:
                self.current_status = "RUN"
            elif angular_z != 0:
                self.current_status = "ROTATION"
        elif self.current_status == "ROTATION":
            if is_stop and angular_z == 0:
                self.current_status = "STOP"
        elif self.current_status == "RUN":
            if is_stop and linear_x == 0:
                self.current_status = "STOP"

    def _cmd_vel_to_rates(self, linear_x: float, angular_z: float) -> Tuple[float, float]:
        """
        Convert CMD_VEL back to joystick rates (reverse of Isaac_Houndbot.compute_cmd_vel).

        This reverses the logic from lines 427-434 in the teleop node.
        """
        linear_rate = 0.0
        angular_rate = 0.0

        if self.current_status == "ROTATION":
            # In rotation mode: linear_vel = 0, angular_vel = max_ang_vel * angular_rate
            if angular_z != 0:
                angular_rate = angular_z / self.config.max_ang_vel
        else:
            # In RUN mode: need to account for angular correction
            if linear_x != 0:
                # Reverse: linear_vel = max_linear_vel * linear_rate
                if linear_x > 0:
                    linear_rate = linear_x / self.config.max_forward_lin_vel
                else:
                    # Handle backward velocity with reduction rate
                    # linear_vel = max(backward_vel_reduction_rate * linear_vel, -max_backward_lin_vel)
                    # So: linear_rate = linear_vel / (max_forward_lin_vel * backward_vel_reduction_rate)
                    linear_rate = linear_x / (self.config.max_forward_lin_vel * self.config.backward_vel_reduction_rate)
                    linear_rate = max(linear_rate, -self.config.max_backward_lin_vel / self.config.max_forward_lin_vel)

            if angular_z != 0:
                # Reverse: angular_vel *= abs(current_twist.linear.x) / max_forward_lin_vel * max_ang_correction
                # So: angular_rate = angular_z / (max_ang_vel * correction_factor)
                if abs(linear_x) > 0.01:  # When running
                    correction_factor = abs(linear_x) / self.config.max_forward_lin_vel * self.config.max_ang_correction
                    angular_rate = angular_z / (self.config.max_ang_vel * correction_factor)
                else:  # When not moving much
                    angular_rate = angular_z / self.config.max_ang_vel

        # Clamp to joystick range
        linear_rate = np.clip(linear_rate, -1.0, 1.0)
        angular_rate = np.clip(angular_rate, -1.0, 1.0)

        return linear_rate, angular_rate
    
    def convert_batch(self, cmd_vel_data: List[Tuple[int, float, float]],
                     remove_duplicates: bool = True) -> List[Dict]:
        """
        Convert a batch of CMD_VEL commands to joystick format.

        Args:
            cmd_vel_data: List of (timestamp, linear_x, angular_z) tuples
            remove_duplicates: If True, remove entries with duplicate timestamps

        Returns:
            List of joystick data dictionaries
        """
        joystick_data = []
        seen_timestamps = set()

        for timestamp, linear_x, angular_z in cmd_vel_data:
            # Skip duplicates if requested
            if remove_duplicates and timestamp in seen_timestamps:
                continue

            joy_msg = self.convert_single(linear_x, angular_z)
            joy_msg['timestamp'] = timestamp
            joystick_data.append(joy_msg)

            if remove_duplicates:
                seen_timestamps.add(timestamp)

        return joystick_data

    def convert_batch_with_rate_limit(self, cmd_vel_data: List[Tuple[int, float, float]],
                                    target_hz: float = 20.0) -> List[Dict]:
        """
        Convert CMD_VEL commands to joystick format with rate limiting.

        Args:
            cmd_vel_data: List of (timestamp, linear_x, angular_z) tuples
            target_hz: Target frequency in Hz (default: 20 Hz)

        Returns:
            List of joystick data dictionaries at specified rate
        """
        if not cmd_vel_data:
            return []

        joystick_data = []
        target_interval_ns = int(1e9 / target_hz)  # Convert Hz to nanoseconds
        last_timestamp = 0

        for timestamp, linear_x, angular_z in cmd_vel_data:
            # Skip if not enough time has passed
            if timestamp - last_timestamp < target_interval_ns:
                continue

            joy_msg = self.convert_single(linear_x, angular_z)
            joy_msg['timestamp'] = timestamp
            joystick_data.append(joy_msg)
            last_timestamp = timestamp

        return joystick_data

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick value."""
        if abs(value) < self.config.deadzone:
            return 0.0
        return value
    
    def get_statistics(self, joystick_data: List[Dict]) -> Dict:
        """Get statistics about the converted joystick data."""
        if not joystick_data:
            return {}
        
        linear_values = [j['linear_axis_value'] for j in joystick_data]
        angular_values = [j['angular_axis_value'] for j in joystick_data]
        
        return {
            'total_commands': len(joystick_data),
            'linear_stats': {
                'min': np.min(linear_values),
                'max': np.max(linear_values),
                'mean': np.mean(linear_values),
                'std': np.std(linear_values),
                'zero_count': sum(1 for v in linear_values if abs(v) < 0.01)
            },
            'angular_stats': {
                'min': np.min(angular_values),
                'max': np.max(angular_values),
                'mean': np.mean(angular_values),
                'std': np.std(angular_values),
                'zero_count': sum(1 for v in angular_values if abs(v) < 0.01)
            },
            'movement_patterns': self._analyze_movement_patterns(joystick_data)
        }
    
    def _analyze_movement_patterns(self, joystick_data: List[Dict]) -> Dict:
        """Analyze common movement patterns in the joystick data."""
        patterns = {
            'forward_only': 0,
            'backward_only': 0,
            'turn_left_only': 0,
            'turn_right_only': 0,
            'forward_left': 0,
            'forward_right': 0,
            'backward_left': 0,
            'backward_right': 0,
            'stationary': 0
        }
        
        for joy in joystick_data:
            linear = joy['linear_axis_value']
            angular = joy['angular_axis_value']
            
            # Define thresholds
            linear_thresh = 0.1
            angular_thresh = 0.1
            
            if abs(linear) < linear_thresh and abs(angular) < angular_thresh:
                patterns['stationary'] += 1
            elif abs(angular) < angular_thresh:  # Pure linear motion
                if linear > linear_thresh:
                    patterns['forward_only'] += 1
                elif linear < -linear_thresh:
                    patterns['backward_only'] += 1
            elif abs(linear) < linear_thresh:  # Pure rotation
                if angular > angular_thresh:
                    patterns['turn_left_only'] += 1
                elif angular < -angular_thresh:
                    patterns['turn_right_only'] += 1
            else:  # Combined motion
                if linear > linear_thresh and angular > angular_thresh:
                    patterns['forward_left'] += 1
                elif linear > linear_thresh and angular < -angular_thresh:
                    patterns['forward_right'] += 1
                elif linear < -linear_thresh and angular > angular_thresh:
                    patterns['backward_left'] += 1
                elif linear < -linear_thresh and angular < -angular_thresh:
                    patterns['backward_right'] += 1
        
        return patterns


def create_isaac_houndbot_config() -> IsaacHoundbotJoystickConfig:
    """Create Isaac_Houndbot joystick config based on actual teleop implementation."""
    return IsaacHoundbotJoystickConfig()


def demo_conversion():
    """Demonstrate the conversion with sample data."""
    
    # Sample CMD_VEL data (timestamp, linear_x, angular_z)
    sample_data = [
        (1000000000, 0.0, 0.0),      # Stationary
        (1001000000, 1.0, 0.0),      # Forward
        (1002000000, -0.5, 0.0),     # Backward
        (1003000000, 0.0, 1.5),      # Turn left
        (1004000000, 0.0, -1.5),     # Turn right
        (1005000000, 0.8, 0.8),      # Forward + left
        (1006000000, 0.8, -0.8),     # Forward + right
        (1007000000, -0.5, 0.5),     # Backward + left
    ]
    
    # Create converter with Isaac_Houndbot configuration
    config = create_isaac_houndbot_config()
    converter = IsaacHoundbotCmdVelToJoystick(config)
    
    print("CMD_VEL to Joystick Conversion Demo")
    print("=" * 50)
    
    # Convert data
    joystick_data = converter.convert_batch(sample_data)
    
    # Show conversions
    print("Conversions:")
    for i, (original, converted) in enumerate(zip(sample_data, joystick_data)):
        _, linear_x, angular_z = original
        linear_axis = converted['linear_axis_value']
        angular_axis = converted['angular_axis_value']
        status = converted['robot_status']
        print(f"{i+1}. CMD_VEL({linear_x:5.1f}, {angular_z:5.1f}) -> "
              f"Joy L/R={angular_axis:6.2f}, F/B={linear_axis:6.2f} [{status}]")
    
    # Show statistics
    stats = converter.get_statistics(joystick_data)
    print(f"\nStatistics:")
    print(f"Total commands: {stats['total_commands']}")
    print(f"Linear axis range: {stats['linear_stats']['min']:.2f} to {stats['linear_stats']['max']:.2f}")
    print(f"Angular axis range: {stats['angular_stats']['min']:.2f} to {stats['angular_stats']['max']:.2f}")
    
    print(f"\nMovement patterns:")
    for pattern, count in stats['movement_patterns'].items():
        if count > 0:
            print(f"  {pattern}: {count}")


if __name__ == "__main__":
    demo_conversion()
