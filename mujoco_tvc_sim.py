"""
Athena TVC MuJoCo Simulation

A physics-accurate simulation using MuJoCo for the Athena thrust vector control system.
Integrates with the existing TVC simulation framework while providing enhanced physics
and visualization capabilities.

Author: TVC Simulation Team
"""

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
import time
import threading
from dataclasses import dataclass

# Import existing TVC components
from tvc_sim import TVCConfiguration, TVCController, QuaternionMath


class MuJoCoTVCSimulation:
    """MuJoCo-based TVC simulation with realistic physics"""
    
    def __init__(self, model_path: str = "athena_tvc.xml", config: TVCConfiguration = None):
        """Initialize MuJoCo TVC simulation
        
        Args:
            model_path: Path to MuJoCo XML model file
            config: TVC configuration (uses default if None)
        """
        self.config = config or TVCConfiguration()
        
        # Load MuJoCo model
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {model_path}: {e}")
        
        # Initialize simulation state
        self.time = 0.0
        self.thrust_magnitude = 0.0
        self.viewer = None
        self.viewer_thread = None
        self.running = False
        
        # Control system
        self.controller = TVCController()
        
        # Get important model indices
        self._setup_model_indices()
        
        # Data logging
        self.time_history = []
        self.gimbal_history = []
        self.actuator_history = []
        self.attitude_history = []
        self.position_history = []
        self.thrust_history = []
        
        print(f"MuJoCo TVC Simulation initialized")
        print(f"Model: {self.model.nq} DOF, {self.model.nu} actuators, {self.model.nsensor} sensors")
    
    def _setup_model_indices(self):
        """Get indices for important model elements"""
        # Joint indices
        self.joint_indices = {
            'gimbal_pitch': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'gimbal_pitch'),
            'gimbal_yaw': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'gimbal_yaw'),
            'actuator_0': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'actuator_0'),
            'actuator_1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'actuator_1'),
            'actuator_2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'actuator_2'),
        }
        
        # Actuator indices
        self.actuator_indices = {
            'actuator_ctrl_0': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator_ctrl_0'),
            'actuator_ctrl_1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator_ctrl_1'),
            'actuator_ctrl_2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator_ctrl_2'),
        }
        
        # Body indices
        self.body_indices = {
            'rocket_body': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rocket_body'),
            'engine': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'engine'),
        }
        
        # Site indices (for thrust application)
        self.site_indices = {
            'thrust_point': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'thrust_point'),
        }
        
        # Sensor indices
        self.sensor_indices = {}
        sensor_names = [
            'gimbal_pitch_pos', 'gimbal_yaw_pos',
            'gimbal_pitch_vel', 'gimbal_yaw_vel',
            'actuator_0_pos', 'actuator_1_pos', 'actuator_2_pos',
            'rocket_orientation', 'rocket_position', 'rocket_velocity', 'rocket_angular_velocity',
            'thrust_point_pos', 'thrust_point_orientation'
        ]
        
        for name in sensor_names:
            try:
                self.sensor_indices[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            except:
                print(f"Warning: Sensor '{name}' not found in model")
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        self.time = 0.0
        self.thrust_magnitude = 0.0
        
        # Set initial actuator positions to neutral
        neutral_length = (self.config.actuator_min_length + self.config.actuator_max_length) / 2
        for i in range(3):
            actuator_idx = self.actuator_indices[f'actuator_ctrl_{i}']
            self.data.ctrl[actuator_idx] = neutral_length
        
        # Clear history
        self.time_history.clear()
        self.gimbal_history.clear()
        self.actuator_history.clear()
        self.attitude_history.clear()
        self.position_history.clear()
        self.thrust_history.clear()
        
        print("Simulation reset to initial state")
    
    def get_gimbal_angles(self) -> Tuple[float, float]:
        """Get current gimbal pitch and yaw angles"""
        pitch_idx = self.joint_indices['gimbal_pitch']
        yaw_idx = self.joint_indices['gimbal_yaw']
        
        pitch = self.data.qpos[pitch_idx]
        yaw = self.data.qpos[yaw_idx]
        
        return pitch, yaw
    
    def get_actuator_positions(self) -> List[float]:
        """Get current actuator positions"""
        positions = []
        for i in range(3):
            joint_idx = self.joint_indices[f'actuator_{i}']
            positions.append(self.data.qpos[joint_idx])
        return positions
    
    def get_rocket_state(self) -> dict:
        """Get complete rocket state information"""
        rocket_body_idx = self.body_indices['rocket_body']
        
        # Position and orientation
        pos = self.data.xpos[rocket_body_idx].copy()
        quat = self.data.xquat[rocket_body_idx].copy()  # [w, x, y, z]
        
        # Velocities
        vel = self.data.cvel[rocket_body_idx][:3].copy()  # Linear velocity
        angvel = self.data.cvel[rocket_body_idx][3:].copy()  # Angular velocity
        
        return {
            'position': pos,
            'orientation': quat,
            'velocity': vel,
            'angular_velocity': angvel,
            'gimbal_angles': self.get_gimbal_angles(),
            'actuator_positions': self.get_actuator_positions()
        }
    
    def set_actuator_targets(self, target_lengths: List[float]):
        """Set target positions for linear actuators"""
        for i, target in enumerate(target_lengths):
            # Clamp to actuator limits
            target = np.clip(target, self.config.actuator_min_length, self.config.actuator_max_length)
            
            actuator_idx = self.actuator_indices[f'actuator_ctrl_{i}']
            self.data.ctrl[actuator_idx] = target
    
    def inverse_kinematics(self, gimbal_x: float, gimbal_y: float) -> List[float]:
        """Compute required actuator lengths for desired gimbal attitude
        
        Uses the same algorithm as the original simulation for consistency.
        """
        target_lengths = []
        
        # Clamp gimbal angles to physical limits
        gimbal_x = np.clip(gimbal_x, -self.config.max_gimbal_angle, self.config.max_gimbal_angle)
        gimbal_y = np.clip(gimbal_y, -self.config.max_gimbal_angle, self.config.max_gimbal_angle)
        
        # Base length (neutral position)
        base_length = (self.config.actuator_min_length + self.config.actuator_max_length) / 2
        
        # Actuator angles (3 actuators at 120-degree spacing)
        actuator_angles = self.config.actuator_angles
        
        for angle in actuator_angles:
            # Compute height change for this actuator position
            height_change = (gimbal_x * np.cos(angle) + gimbal_y * np.sin(angle)) * self.config.gimbal_radius
            
            target_length = base_length + height_change
            target_length = np.clip(target_length, self.config.actuator_min_length, self.config.actuator_max_length)
            target_lengths.append(target_length)
        
        return target_lengths
    
    def apply_thrust(self, thrust_magnitude: float = None):
        """Apply thrust force through the engine nozzle"""
        if thrust_magnitude is not None:
            self.thrust_magnitude = np.clip(thrust_magnitude, 0, self.config.max_thrust)
        
        if self.thrust_magnitude <= 0:
            return
        
        # Get thrust application point and orientation
        site_idx = self.site_indices['thrust_point']
        thrust_pos = self.data.site_xpos[site_idx]
        thrust_mat = self.data.site_xmat[site_idx].reshape(3, 3)
        
        # Thrust direction is along the nozzle axis (negative Z in local coordinates)
        thrust_direction_local = np.array([0, 0, -1])
        thrust_direction_world = thrust_mat @ thrust_direction_local
        
        # Apply thrust force at the thrust point
        thrust_force = thrust_direction_world * self.thrust_magnitude
        
        # Get the body index for force application
        engine_body_idx = self.body_indices['engine']
        
        # Convert world position to body-relative position
        engine_pos = self.data.xpos[engine_body_idx]
        relative_pos = thrust_pos - engine_pos
        
        # Apply force at the specified point
        mujoco.mj_applyFT(
            self.model, self.data,
            thrust_force,      # Force vector
            np.zeros(3),       # Torque vector (thrust is pure force)
            relative_pos,      # Point of application (relative to body)
            engine_body_idx,   # Body index
            self.data.qfrc_applied  # Applied forces array
        )
    
    def step_simulation(self, desired_gimbal: np.ndarray = None, thrust: float = None):
        """Advance simulation by one time step"""
        if desired_gimbal is None:
            desired_gimbal = np.array([0.0, 0.0])
        
        # Get current gimbal attitude
        current_gimbal = np.array(self.get_gimbal_angles())
        
        # Control system - compute required actuator positions
        # Use inverse kinematics directly for now (could use PID on gimbal angles)
        target_lengths = self.inverse_kinematics(desired_gimbal[0], desired_gimbal[1])
        self.set_actuator_targets(target_lengths)
        
        # Apply thrust if specified
        if thrust is not None:
            self.apply_thrust(thrust)
        else:
            self.apply_thrust()
        
        # Step the physics simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update time
        self.time = self.data.time
        
        # Log data
        self._log_data()
    
    def _log_data(self):
        """Log simulation data for analysis"""
        gimbal_angles = self.get_gimbal_angles()
        actuator_positions = self.get_actuator_positions()
        rocket_state = self.get_rocket_state()
        
        self.time_history.append(self.time)
        self.gimbal_history.append(list(gimbal_angles))
        self.actuator_history.append(actuator_positions)
        self.attitude_history.append(rocket_state['orientation'].copy())
        self.position_history.append(rocket_state['position'].copy())
        self.thrust_history.append(self.thrust_magnitude)
    
    def run_simulation(self, duration: float, thrust_profile: Callable = None, 
                      gimbal_commands: Callable = None, realtime: bool = False):
        """Run simulation for specified duration
        
        Args:
            duration: Simulation duration in seconds
            thrust_profile: Function that returns thrust given time
            gimbal_commands: Function that returns desired gimbal angles given time
            realtime: Whether to run in real-time (for visualization)
        """
        print(f"Running MuJoCo simulation for {duration:.2f}s")
        
        start_time = time.time()
        steps = int(duration / self.model.opt.timestep)
        
        for i in range(steps):
            # Update thrust if profile provided
            current_thrust = self.thrust_magnitude
            if thrust_profile is not None:
                if callable(thrust_profile):
                    current_thrust = thrust_profile(self.time)
                else:
                    current_thrust = thrust_profile
            
            # Update gimbal commands if provided
            desired_gimbal = np.array([0.0, 0.0])
            if gimbal_commands is not None:
                if callable(gimbal_commands):
                    desired_gimbal = gimbal_commands(self.time)
                else:
                    desired_gimbal = np.array(gimbal_commands)
            
            # Step simulation
            self.step_simulation(desired_gimbal, current_thrust)
            
            # Real-time control
            if realtime:
                elapsed = time.time() - start_time
                expected_time = i * self.model.opt.timestep
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
            
            # Progress update
            if i % (steps // 10) == 0:
                print(f"Progress: {100*i/steps:.1f}%")
        
        elapsed_time = time.time() - start_time
        print(f"Simulation complete! Elapsed time: {elapsed_time:.2f}s")
        print(f"Simulation rate: {duration/elapsed_time:.1f}x real-time")
    
    def start_viewer(self):
        """Start the MuJoCo viewer in a separate thread"""
        if self.viewer is not None:
            print("Viewer already running")
            return
        
        def viewer_loop():
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                self.running = True
                
                while self.running:
                    # Sync viewer with simulation data
                    viewer.sync()
                    time.sleep(0.01)  # 100 Hz update rate
                
                self.viewer = None
        
        self.viewer_thread = threading.Thread(target=viewer_loop, daemon=True)
        self.viewer_thread.start()
        time.sleep(0.5)  # Give viewer time to start
        print("MuJoCo viewer started")
    
    def stop_viewer(self):
        """Stop the MuJoCo viewer"""
        if self.viewer is not None:
            self.running = False
            if self.viewer_thread:
                self.viewer_thread.join(timeout=2)
            print("MuJoCo viewer stopped")
    
    def run_interactive_simulation(self, duration: float = None):
        """Run an interactive simulation with real-time visualization"""
        self.start_viewer()
        
        try:
            print("Interactive simulation started. Use viewer controls to interact.")
            print("Running basic test pattern...")
            
            # Simple test pattern
            def test_gimbal(t):
                return np.array([
                    0.1 * np.sin(2 * np.pi * 0.5 * t),  # 0.5 Hz pitch
                    0.05 * np.cos(2 * np.pi * 0.3 * t)   # 0.3 Hz yaw
                ])
            
            def test_thrust(t):
                if t < 2:
                    return 30.0
                elif t < 8:
                    return 50.0
                else:
                    return 20.0
            
            # Run simulation
            run_duration = duration if duration else 10.0
            self.run_simulation(
                duration=run_duration,
                thrust_profile=test_thrust,
                gimbal_commands=test_gimbal,
                realtime=True
            )
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            self.stop_viewer()
    
    def plot_results(self):
        """Plot simulation results (compatible with original simulation)"""
        if not self.time_history:
            print("No data to plot - run simulation first")
            return
        
        time_array = np.array(self.time_history)
        gimbal_array = np.array(self.gimbal_history)
        actuator_array = np.array(self.actuator_history)
        position_array = np.array(self.position_history)
        attitude_array = np.array(self.attitude_history)
        thrust_array = np.array(self.thrust_history)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Athena TVC MuJoCo Simulation Results')
        
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
            axes[0,1].plot(time_array, actuator_array[:, i], label=f'Actuator {i+1}')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Length (m)')
        axes[0,1].set_title('Actuator Lengths')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Thrust profile
        axes[0,2].plot(time_array, thrust_array, 'g-')
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Thrust (N)')
        axes[0,2].set_title('Thrust Profile')
        axes[0,2].grid(True)
        
        # Rocket position
        axes[1,0].plot(time_array, position_array[:, 0], 'r-', label='X')
        axes[1,0].plot(time_array, position_array[:, 1], 'g-', label='Y')
        axes[1,0].plot(time_array, position_array[:, 2], 'b-', label='Z')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Position (m)')
        axes[1,0].set_title('Rocket Position')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 3D trajectory
        ax_3d = plt.axes([0.37, 0.05, 0.25, 0.4], projection='3d')
        ax_3d.plot(position_array[:, 0], position_array[:, 1], position_array[:, 2], 'b-')
        ax_3d.scatter(position_array[0, 0], position_array[0, 1], position_array[0, 2], 
                     c='g', s=50, label='Start')
        ax_3d.scatter(position_array[-1, 0], position_array[-1, 1], position_array[-1, 2], 
                     c='r', s=50, label='End')
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Trajectory')
        ax_3d.legend()
        
        # Rocket attitude (quaternion magnitude for simplicity)
        quat_magnitude = np.linalg.norm(attitude_array, axis=1)
        axes[1,2].plot(time_array, quat_magnitude, 'purple')
        axes[1,2].set_xlabel('Time (s)')
        axes[1,2].set_ylabel('Quaternion Norm')
        axes[1,2].set_title('Attitude Stability')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_original(self, original_sim, duration: float = 5.0):
        """Compare MuJoCo simulation results with original simulation"""
        print("Running comparison between MuJoCo and original simulations...")
        
        # Define test conditions
        def test_gimbal(t):
            return np.array([0.1 * np.sin(t), 0.05 * np.cos(t)])
        
        def test_thrust(t):
            return 50.0 if t < duration/2 else 25.0
        
        # Run MuJoCo simulation
        self.reset_simulation()
        self.run_simulation(duration, test_thrust, test_gimbal)
        mujoco_gimbal = np.array(self.gimbal_history)
        mujoco_actuators = np.array(self.actuator_history)
        
        # Run original simulation
        original_sim.time = 0.0
        original_sim.time_history.clear()
        original_sim.gimbal_history.clear()
        original_sim.actuator_history.clear()
        
        original_sim.run_simulation(duration, test_thrust, test_gimbal)
        original_gimbal = np.array(original_sim.gimbal_history)
        original_actuators = np.array(original_sim.actuator_history)
        
        # Compare results
        time_array = np.array(self.time_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('MuJoCo vs Original Simulation Comparison')
        
        # Gimbal pitch comparison
        axes[0,0].plot(time_array, np.degrees(mujoco_gimbal[:, 0]), 'r-', label='MuJoCo')
        axes[0,0].plot(np.array(original_sim.time_history), 
                      np.degrees(original_gimbal[:, 0]), 'b--', label='Original')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Pitch Angle (deg)')
        axes[0,0].set_title('Gimbal Pitch Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Gimbal yaw comparison
        axes[0,1].plot(time_array, np.degrees(mujoco_gimbal[:, 1]), 'r-', label='MuJoCo')
        axes[0,1].plot(np.array(original_sim.time_history), 
                      np.degrees(original_gimbal[:, 1]), 'b--', label='Original')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Yaw Angle (deg)')
        axes[0,1].set_title('Gimbal Yaw Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Actuator comparison (first actuator)
        axes[1,0].plot(time_array, mujoco_actuators[:, 0], 'r-', label='MuJoCo')
        axes[1,0].plot(np.array(original_sim.time_history), 
                      original_actuators[:, 0], 'b--', label='Original')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Actuator 1 Length (m)')
        axes[1,0].set_title('Actuator Length Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Error analysis
        # Interpolate to common time base for error calculation
        common_time = np.linspace(0, duration, min(len(time_array), len(original_sim.time_history)))
        mujoco_pitch_interp = np.interp(common_time, time_array, mujoco_gimbal[:, 0])
        original_pitch_interp = np.interp(common_time, original_sim.time_history, original_gimbal[:, 0])
        
        pitch_error = np.abs(mujoco_pitch_interp - original_pitch_interp)
        axes[1,1].plot(common_time, np.degrees(pitch_error), 'g-')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Pitch Error (deg)')
        axes[1,1].set_title('Simulation Differences')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        mean_error = np.mean(pitch_error)
        max_error = np.max(pitch_error)
        print(f"Pitch angle comparison:")
        print(f"  Mean error: {np.degrees(mean_error):.4f} deg")
        print(f"  Max error:  {np.degrees(max_error):.4f} deg")


def example_mujoco_simulation():
    """Run an example MuJoCo simulation"""
    print("=== Athena TVC MuJoCo Simulation Demo ===")
    
    try:
        # Create MuJoCo simulation
        sim = MuJoCoTVCSimulation()
        
        # Define test profiles
        def thrust_profile(t):
            if t < 2.0:
                return 40.0  # Initial thrust
            elif t < 6.0:
                return 60.0  # Higher thrust
            else:
                return 30.0  # Reduced thrust
        
        def gimbal_commands(t):
            return np.array([
                0.12 * np.sin(2 * np.pi * 0.4 * t),  # Pitch oscillation
                0.08 * np.cos(2 * np.pi * 0.6 * t)   # Yaw oscillation
            ])
        
        # Run simulation
        sim.run_simulation(
            duration=8.0,
            thrust_profile=thrust_profile,
            gimbal_commands=gimbal_commands
        )
        
        # Plot results
        sim.plot_results()
        
        # Print final state
        final_state = sim.get_rocket_state()
        print(f"\nFinal State:")
        print(f"Time: {sim.time:.3f}s")
        print(f"Position: {final_state['position']}")
        print(f"Gimbal angles: {np.degrees(final_state['gimbal_angles'])} deg")
        print(f"Actuator positions: {final_state['actuator_positions']}")
        
        return sim
        
    except Exception as e:
        print(f"Error in MuJoCo simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run example simulation
    sim = example_mujoco_simulation()
    
    if sim is not None:
        # Optionally run interactive simulation
        user_input = input("\nRun interactive simulation with 3D viewer? (y/n): ")
        if user_input.lower() == 'y':
            sim.reset_simulation()
            sim.run_interactive_simulation(duration=10.0) 