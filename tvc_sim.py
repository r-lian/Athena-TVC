"""
Athena Thrust Vector Control (TVC) Physics Simulation

A comprehensive physics-based simulation for rocket engine thrust vector control
with inverse kinematics, gimbal mechanics, and linear actuator modeling.

Author: TVC Simulation Team
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TVCConfiguration:
    """Configuration parameters for the TVC system"""
    # Physical dimensions (meters)
    gimbal_radius: float = 0.05  # Distance from pivot to actuator mounting points
    actuator_min_length: float = 0.08  # Minimum actuator length
    actuator_max_length: float = 0.12  # Maximum actuator length
    engine_mass: float = 2.0  # Engine mass (kg)
    
    # Gimbal geometry
    actuator_angles: np.ndarray = None  # Angles of actuator mounting points
    max_gimbal_angle: float = np.radians(15)  # Maximum gimbal deflection
    
    # Control parameters
    max_thrust: float = 100.0  # Maximum thrust (N)
    thrust_response_time: float = 0.1  # Thrust response time constant
    
    def __post_init__(self):
        if self.actuator_angles is None:
            # Default to 3 actuators at 120-degree spacing
            self.actuator_angles = np.array([0, 2*np.pi/3, 4*np.pi/3])


class QuaternionMath:
    """Quaternion mathematics for 3D rotations"""
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion from axis-angle representation"""
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        return np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ])
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion"""
        v_quat = np.array([0, v[0], v[1], v[2]])
        rotated = QuaternionMath.multiply(
            QuaternionMath.multiply(q, v_quat),
            QuaternionMath.conjugate(q)
        )
        return rotated[1:4]
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])


class LinearActuator:
    """Model of a linear actuator for TVC"""
    
    def __init__(self, position_angle: float, radius: float, 
                 min_length: float, max_length: float):
        self.position_angle = position_angle
        self.radius = radius
        self.min_length = min_length
        self.max_length = max_length
        self.current_length = (min_length + max_length) / 2
        self.target_length = self.current_length
        
        # Actuator dynamics
        self.velocity = 0.0
        self.max_velocity = 0.05  # m/s
        self.max_acceleration = 0.5  # m/s^2
        
        # Fixed mounting point on base
        self.base_point = np.array([
            radius * np.cos(position_angle),
            radius * np.sin(position_angle),
            0
        ])
    
    def update(self, dt: float, target_length: float):
        """Update actuator position with dynamics"""
        self.target_length = np.clip(target_length, self.min_length, self.max_length)
        
        # Simple PD control for actuator positioning
        error = self.target_length - self.current_length
        desired_velocity = 5.0 * error  # Proportional gain
        velocity_error = desired_velocity - self.velocity
        
        # Apply acceleration limits
        acceleration = np.clip(velocity_error / dt, 
                             -self.max_acceleration, self.max_acceleration)
        self.velocity += acceleration * dt
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        
        self.current_length += self.velocity * dt
        self.current_length = np.clip(self.current_length, 
                                    self.min_length, self.max_length)


class GimbalKinematics:
    """Handles gimbal kinematics and inverse kinematics"""
    
    def __init__(self, config: TVCConfiguration):
        self.config = config
        self.actuators = []
        
        # Create actuators
        for angle in config.actuator_angles:
            actuator = LinearActuator(
                position_angle=angle,
                radius=config.gimbal_radius,
                min_length=config.actuator_min_length,
                max_length=config.actuator_max_length
            )
            self.actuators.append(actuator)
    
    def compute_gimbal_attitude(self) -> Tuple[float, float]:
        """Compute gimbal attitude from actuator lengths (forward kinematics)"""
        # This is a simplified calculation for demonstration
        # In practice, this would solve the forward kinematics equations
        
        # For 3-actuator system, compute average height and tilt
        lengths = np.array([act.current_length for act in self.actuators])
        mean_length = np.mean(lengths)
        
        # Compute tilt components
        gimbal_x = 0
        gimbal_y = 0
        
        for i, actuator in enumerate(self.actuators):
            angle = actuator.position_angle
            length_diff = actuator.current_length - mean_length
            gimbal_x += length_diff * np.cos(angle)
            gimbal_y += length_diff * np.sin(angle)
        
        # Scale to get actual gimbal angles
        scale_factor = 1.0 / self.config.gimbal_radius
        gimbal_x *= scale_factor
        gimbal_y *= scale_factor
        
        return gimbal_x, gimbal_y
    
    def inverse_kinematics(self, gimbal_x: float, gimbal_y: float) -> List[float]:
        """Compute required actuator lengths for desired gimbal attitude"""
        target_lengths = []
        
        # Clamp gimbal angles to physical limits
        gimbal_x = np.clip(gimbal_x, -self.config.max_gimbal_angle, 
                          self.config.max_gimbal_angle)
        gimbal_y = np.clip(gimbal_y, -self.config.max_gimbal_angle, 
                          self.config.max_gimbal_angle)
        
        # Base length (neutral position)
        base_length = (self.config.actuator_min_length + 
                      self.config.actuator_max_length) / 2
        
        for actuator in self.actuators:
            # Compute height change for this actuator position
            angle = actuator.position_angle
            height_change = (gimbal_x * np.cos(angle) + 
                           gimbal_y * np.sin(angle)) * self.config.gimbal_radius
            
            target_length = base_length + height_change
            target_length = np.clip(target_length, 
                                  self.config.actuator_min_length,
                                  self.config.actuator_max_length)
            target_lengths.append(target_length)
        
        return target_lengths
    
    def update_actuators(self, dt: float, gimbal_x: float, gimbal_y: float):
        """Update all actuators to achieve desired gimbal attitude"""
        target_lengths = self.inverse_kinematics(gimbal_x, gimbal_y)
        
        for actuator, target in zip(self.actuators, target_lengths):
            actuator.update(dt, target)


class RocketDynamics:
    """Simplified rocket dynamics for TVC simulation"""
    
    def __init__(self, mass: float):
        self.mass = mass
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion (w,x,y,z)
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        
        # Moments of inertia (simplified)
        self.inertia = np.diag([0.1, 0.1, 0.05])
        
    def apply_forces_and_torques(self, force: np.ndarray, torque: np.ndarray, dt: float):
        """Apply forces and torques to update dynamics"""
        # Linear dynamics
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Angular dynamics (simplified)
        angular_acceleration = np.linalg.solve(self.inertia, torque)
        self.angular_velocity += angular_acceleration * dt
        
        # Update attitude (simplified quaternion integration)
        if np.linalg.norm(self.angular_velocity) > 1e-10:
            axis = self.angular_velocity / np.linalg.norm(self.angular_velocity)
            angle = np.linalg.norm(self.angular_velocity) * dt
            dq = QuaternionMath.from_axis_angle(axis, angle)
            self.attitude = QuaternionMath.multiply(self.attitude, dq)
            # Normalize quaternion
            self.attitude /= np.linalg.norm(self.attitude)


class TVCController:
    """PID controller for thrust vector control"""
    
    def __init__(self):
        # PID gains
        self.kp = np.array([2.0, 2.0])  # Proportional gains for x,y
        self.ki = np.array([0.1, 0.1])  # Integral gains
        self.kd = np.array([0.5, 0.5])  # Derivative gains
        
        # Controller state
        self.integral_error = np.array([0.0, 0.0])
        self.previous_error = np.array([0.0, 0.0])
        
    def compute_control(self, desired_attitude: np.ndarray, 
                       current_attitude: np.ndarray, dt: float) -> np.ndarray:
        """Compute control commands for gimbal angles"""
        # Error computation
        error = desired_attitude - current_attitude
        
        # Integral term
        self.integral_error += error * dt
        
        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        
        # PID control law
        control = (self.kp * error + 
                  self.ki * self.integral_error + 
                  self.kd * derivative)
        
        self.previous_error = error.copy()
        
        return control


class AthenaTVCSimulation:
    """Main simulation class for Athena TVC platform"""
    
    def __init__(self, config: TVCConfiguration = None):
        self.config = config or TVCConfiguration()
        
        # Initialize subsystems
        self.gimbal = GimbalKinematics(self.config)
        self.rocket = RocketDynamics(self.config.engine_mass)
        self.controller = TVCController()
        
        # Simulation state
        self.time = 0.0
        self.dt = 0.001  # 1ms time step
        self.thrust_magnitude = 0.0
        
        # Data logging
        self.time_history = []
        self.gimbal_history = []
        self.actuator_history = []
        self.attitude_history = []
        
    def set_thrust(self, thrust: float):
        """Set thrust magnitude"""
        self.thrust_magnitude = np.clip(thrust, 0, self.config.max_thrust)
    
    def step_simulation(self, desired_gimbal: np.ndarray = None):
        """Advance simulation by one time step"""
        if desired_gimbal is None:
            desired_gimbal = np.array([0.0, 0.0])
        
        # Get current gimbal attitude
        current_gimbal = np.array(self.gimbal.compute_gimbal_attitude())
        
        # Control system
        control_output = self.controller.compute_control(
            desired_gimbal, current_gimbal, self.dt
        )
        
        # Update gimbal actuators
        self.gimbal.update_actuators(self.dt, control_output[0], control_output[1])
        
        # Compute thrust vector in body frame
        gimbal_x, gimbal_y = self.gimbal.compute_gimbal_attitude()
        thrust_direction = np.array([
            np.sin(gimbal_x),
            np.sin(gimbal_y),
            np.cos(gimbal_x) * np.cos(gimbal_y)
        ])
        thrust_force = thrust_direction * self.thrust_magnitude
        
        # Transform thrust to world frame
        world_thrust = QuaternionMath.rotate_vector(self.rocket.attitude, thrust_force)
        
        # Compute torques due to gimbal deflection
        # Simplified: assume engine is offset from CG
        engine_offset = np.array([0, 0, -0.5])  # 0.5m below CG
        torque = np.cross(engine_offset, thrust_force)
        
        # Add gravity
        gravity = np.array([0, 0, -9.81 * self.rocket.mass])
        total_force = world_thrust + gravity
        
        # Update rocket dynamics
        self.rocket.apply_forces_and_torques(total_force, torque, self.dt)
        
        # Log data
        self.time += self.dt
        self.time_history.append(self.time)
        self.gimbal_history.append([gimbal_x, gimbal_y])
        self.actuator_history.append([act.current_length for act in self.gimbal.actuators])
        self.attitude_history.append(self.rocket.attitude.copy())
    
    def run_simulation(self, duration: float, thrust_profile=None, 
                      gimbal_commands=None):
        """Run simulation for specified duration"""
        steps = int(duration / self.dt)
        
        print(f"Running simulation for {duration:.2f}s ({steps} steps)")
        
        for i in range(steps):
            # Update thrust if profile provided
            if thrust_profile is not None:
                if callable(thrust_profile):
                    self.set_thrust(thrust_profile(self.time))
                else:
                    self.set_thrust(thrust_profile)
            
            # Update gimbal commands if provided
            desired_gimbal = np.array([0.0, 0.0])
            if gimbal_commands is not None:
                if callable(gimbal_commands):
                    desired_gimbal = gimbal_commands(self.time)
                else:
                    desired_gimbal = np.array(gimbal_commands)
            
            self.step_simulation(desired_gimbal)
            
            # Progress update
            if i % (steps // 10) == 0:
                print(f"Progress: {100*i/steps:.1f}%")
        
        print("Simulation complete!")
    
    def plot_results(self):
        """Plot simulation results"""
        if not self.time_history:
            print("No data to plot - run simulation first")
            return
        
        time_array = np.array(self.time_history)
        gimbal_array = np.array(self.gimbal_history)
        actuator_array = np.array(self.actuator_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Athena TVC Simulation Results')
        
        # Gimbal angles
        axes[0,0].plot(time_array, np.degrees(gimbal_array[:, 0]), 'r-', label='Pitch')
        axes[0,0].plot(time_array, np.degrees(gimbal_array[:, 1]), 'b-', label='Yaw')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Gimbal Angle (deg)')
        axes[0,0].set_title('Gimbal Angles')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Actuator lengths
        for i in range(actuator_array.shape[1]):
            axes[0,1].plot(time_array, actuator_array[:, i], 
                          label=f'Actuator {i+1}')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Length (m)')
        axes[0,1].set_title('Actuator Lengths')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Rocket position
        if hasattr(self.rocket, 'position'):
            position_history = []
            for i in range(len(self.time_history)):
                # We need to track position history
                pass
        
        # 3D trajectory would go here
        axes[1,0].text(0.5, 0.5, 'Rocket Trajectory\n(3D plot placeholder)', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Rocket Trajectory')
        
        # Thrust vector visualization
        axes[1,1].text(0.5, 0.5, 'Thrust Vector\n(visualization placeholder)', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Thrust Vector')
        
        plt.tight_layout()
        plt.show()


def example_simulation():
    """Run an example simulation with test commands"""
    print("=== Athena TVC Simulation Demo ===")
    
    # Create configuration
    config = TVCConfiguration()
    print(f"Configuration: {config.actuator_angles.shape[0]} actuators, "
          f"max angle: {np.degrees(config.max_gimbal_angle):.1f}°")
    
    # Create simulation
    sim = AthenaTVCSimulation(config)
    
    # Define test thrust profile
    def thrust_profile(t):
        if t < 1.0:
            return 50.0  # Constant thrust
        elif t < 3.0:
            return 30.0
        else:
            return 0.0
    
    # Define gimbal command profile
    def gimbal_commands(t):
        # Sinusoidal gimbal commands for testing
        return np.array([
            0.1 * np.sin(2 * np.pi * 0.5 * t),  # 0.5 Hz pitch oscillation
            0.05 * np.cos(2 * np.pi * 0.3 * t)   # 0.3 Hz yaw oscillation
        ])
    
    # Run simulation
    sim.run_simulation(
        duration=5.0,
        thrust_profile=thrust_profile,
        gimbal_commands=gimbal_commands
    )
    
    # Plot results
    sim.plot_results()
    
    # Print final state
    print(f"\nFinal State:")
    print(f"Time: {sim.time:.3f}s")
    print(f"Gimbal angles: {np.degrees(sim.gimbal.compute_gimbal_attitude())} deg")
    print(f"Actuator lengths: {[f'{act.current_length:.4f}' for act in sim.gimbal.actuators]} m")


if __name__ == "__main__":
    # Run example simulation
    example_simulation()
