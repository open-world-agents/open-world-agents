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
class JoystickConfig:
    """Configuration for joystick conversion."""
    # Maximum velocities (Isaac_Houndbot configuration)
    max_linear_vel: float = 15.0  # m/s (from Isaac_Houndbot.yml)
    max_angular_vel: float = 10.0 # rad/s (from Isaac_Houndbot.yml)
    
    # Joystick axis ranges (typical gamepad: -1.0 to 1.0)
    axis_min: float = -1.0
    axis_max: float = 1.0
    
    # Deadzone for joystick (values below this are considered zero)
    # Reduced for Isaac_Houndbot's high velocity range
    deadzone: float = 0.02
    
    # Axis mapping (typical gamepad layout)
    linear_axis: int = 1   # Left stick Y-axis (forward/backward)
    angular_axis: int = 0  # Left stick X-axis (left/right turn)
    
    # Invert axes if needed (depends on joystick orientation)
    invert_linear: bool = False  # True if up on stick should be negative
    invert_angular: bool = False # True if right on stick should be negative


class CmdVelToJoystick:
    """Convert CMD_VEL commands back to joystick input format."""
    
    def __init__(self, config: Optional[JoystickConfig] = None):
        self.config = config or JoystickConfig()
    
    def convert_single(self, linear_x: float, angular_z: float) -> Dict[str, float]:
        """
        Convert a single CMD_VEL command to joystick axes.
        
        Args:
            linear_x: Linear velocity in m/s
            angular_z: Angular velocity in rad/s
        
        Returns:
            Dictionary with joystick axis values
        """
        # Normalize velocities to joystick range
        linear_normalized = self._normalize_velocity(
            linear_x, self.config.max_linear_vel
        )
        angular_normalized = self._normalize_velocity(
            angular_z, self.config.max_angular_vel
        )
        
        # Apply inversions if configured
        if self.config.invert_linear:
            linear_normalized = -linear_normalized
        if self.config.invert_angular:
            angular_normalized = -angular_normalized
        
        # Apply deadzone
        linear_normalized = self._apply_deadzone(linear_normalized)
        angular_normalized = self._apply_deadzone(angular_normalized)
        
        # Create joystick message format (typical 4-axis gamepad)
        axes = [0.0, 0.0, 0.0, 0.0]  # [left_x, left_y, right_x, right_y]
        axes[self.config.angular_axis] = angular_normalized
        axes[self.config.linear_axis] = linear_normalized
        
        return {
            'axes': axes,
            'buttons': [0] * 8,  # Typical gamepad has 8 buttons
            'linear_axis_value': linear_normalized,
            'angular_axis_value': angular_normalized,
            'original_linear_x': linear_x,
            'original_angular_z': angular_z
        }
    
    def convert_batch(self, cmd_vel_data: List[Tuple[int, float, float]]) -> List[Dict]:
        """
        Convert a batch of CMD_VEL commands to joystick format.
        
        Args:
            cmd_vel_data: List of (timestamp, linear_x, angular_z) tuples
        
        Returns:
            List of joystick data dictionaries
        """
        joystick_data = []
        
        for timestamp, linear_x, angular_z in cmd_vel_data:
            joy_msg = self.convert_single(linear_x, angular_z)
            joy_msg['timestamp'] = timestamp
            joystick_data.append(joy_msg)
        
        return joystick_data
    
    def _normalize_velocity(self, velocity: float, max_velocity: float) -> float:
        """Normalize velocity to joystick axis range."""
        if max_velocity == 0:
            return 0.0
        
        normalized = velocity / max_velocity
        # Clamp to joystick range
        return np.clip(normalized, self.config.axis_min, self.config.axis_max)
    
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


def create_joystick_config_from_robot_params(max_v: float = 15.0, max_w: float = 10.0) -> JoystickConfig:
    """Create joystick config based on Isaac_Houndbot parameters."""
    return JoystickConfig(
        max_linear_vel=max_v,
        max_angular_vel=max_w,
        # Standard gamepad configuration
        linear_axis=1,    # Left stick Y
        angular_axis=0,   # Left stick X
        invert_linear=False,  # Adjust based on your setup
        invert_angular=False,
        deadzone=0.02
    )


def create_isaac_houndbot_config() -> JoystickConfig:
    """Create joystick config specifically for Isaac_Houndbot from YAML file."""
    config_path = Path("/mnt/home/jyjung/sketchdrive/projects/drive_evaluation/src/drive_evaluation/agent/pd_controller/Isaac_Houndbot.yml")

    # Default values from the YAML file
    max_v = 15.0  # m/s
    max_w = 10.0  # rad/s

    # Try to load from YAML if available
    if HAS_YAML and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                max_v = float(config_data.get('max_v', 15.0))
                max_w = float(config_data.get('max_w', 10.0))
                print(f"Loaded Isaac_Houndbot config: max_v={max_v}, max_w={max_w}")
        except Exception as e:
            print(f"Warning: Could not load Isaac_Houndbot config: {e}")
            print(f"Using default values: max_v={max_v}, max_w={max_w}")
    else:
        print(f"Using hardcoded Isaac_Houndbot values: max_v={max_v}, max_w={max_w}")

    return JoystickConfig(
        max_linear_vel=max_v,
        max_angular_vel=max_w,
        linear_axis=1,    # Left stick Y (forward/backward)
        angular_axis=0,   # Left stick X (left/right)
        invert_linear=False,
        invert_angular=False,
        deadzone=0.02
    )


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
    converter = CmdVelToJoystick(config)
    
    print("CMD_VEL to Joystick Conversion Demo")
    print("=" * 50)
    
    # Convert data
    joystick_data = converter.convert_batch(sample_data)
    
    # Show conversions
    print("Conversions:")
    for i, (original, converted) in enumerate(zip(sample_data, joystick_data)):
        timestamp, linear_x, angular_z = original
        axes = converted['axes']
        print(f"{i+1}. CMD_VEL({linear_x:5.1f}, {angular_z:5.1f}) -> "
              f"Joy axes[{axes[0]:5.2f}, {axes[1]:5.2f}, {axes[2]:5.2f}, {axes[3]:5.2f}]")
    
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
