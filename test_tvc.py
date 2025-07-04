"""
Test script for Athena TVC Simulation

Demonstrates various capabilities and validates system performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from tvc_sim import AthenaTVCSimulation, TVCConfiguration, QuaternionMath


def test_inverse_kinematics():
    """Test inverse kinematics accuracy"""
    print("=== Testing Inverse Kinematics ===")
    
    config = TVCConfiguration()
    sim = AthenaTVCSimulation(config)
    
    # Test various gimbal angles
    test_angles = [
        (0.0, 0.0),      # Neutral
        (0.1, 0.0),      # Pitch only
        (0.0, 0.1),      # Yaw only
        (0.1, 0.1),      # Combined
        (0.15, 0.15),    # Near maximum
    ]
    
    print("Gimbal Angle -> Actuator Lengths")
    print("--------------------------------")
    
    for gimbal_x, gimbal_y in test_angles:
        target_lengths = sim.gimbal.inverse_kinematics(gimbal_x, gimbal_y)
        
        # Set actuators to computed lengths
        for actuator, length in zip(sim.gimbal.actuators, target_lengths):
            actuator.current_length = length
        
        # Compute forward kinematics
        computed_x, computed_y = sim.gimbal.compute_gimbal_attitude()
        
        error_x = abs(gimbal_x - computed_x)
        error_y = abs(gimbal_y - computed_y)
        
        print(f"({gimbal_x:6.3f}, {gimbal_y:6.3f}) -> "
              f"[{target_lengths[0]:.4f}, {target_lengths[1]:.4f}, {target_lengths[2]:.4f}] -> "
              f"({computed_x:6.3f}, {computed_y:6.3f}) "
              f"Error: ({error_x:.5f}, {error_y:.5f})")


def test_actuator_dynamics():
    """Test actuator response and dynamics"""
    print("\n=== Testing Actuator Dynamics ===")
    
    config = TVCConfiguration()
    sim = AthenaTVCSimulation(config)
    
    # Step response test
    actuator = sim.gimbal.actuators[0]
    initial_length = actuator.current_length
    target = initial_length + 0.02  # 2cm step
    
    time_data = []
    length_data = []
    velocity_data = []
    
    dt = 0.001
    for i in range(500):  # 0.5 seconds
        time_data.append(i * dt)
        length_data.append(actuator.current_length)
        velocity_data.append(actuator.velocity)
        
        actuator.update(dt, target)
    
    # Calculate settling time and overshoot
    final_value = length_data[-1]
    settling_tolerance = 0.02 * abs(target - initial_length)  # 2% tolerance
    
    settling_time = None
    for i in range(len(length_data) - 1, -1, -1):
        if abs(length_data[i] - final_value) > settling_tolerance:
            settling_time = time_data[i + 1] if i + 1 < len(time_data) else time_data[-1]
            break
    
    max_overshoot = max(length_data) - target if max(length_data) > target else 0
    
    print(f"Step Response Characteristics:")
    print(f"  Target: {target:.4f}m")
    print(f"  Final: {final_value:.4f}m")
    print(f"  Settling time: {settling_time:.3f}s")
    print(f"  Overshoot: {max_overshoot:.5f}m")
    print(f"  Steady-state error: {abs(target - final_value):.5f}m")


def test_quaternion_math():
    """Test quaternion mathematics"""
    print("\n=== Testing Quaternion Mathematics ===")
    
    # Test identity rotation
    q_identity = np.array([1, 0, 0, 0])
    test_vector = np.array([1, 2, 3])
    rotated = QuaternionMath.rotate_vector(q_identity, test_vector)
    
    print(f"Identity rotation test:")
    print(f"  Original: {test_vector}")
    print(f"  Rotated:  {rotated}")
    print(f"  Error:    {np.linalg.norm(test_vector - rotated):.10f}")
    
    # Test 90-degree rotation about z-axis
    q_90z = QuaternionMath.from_axis_angle(np.array([0, 0, 1]), np.pi/2)
    test_x = np.array([1, 0, 0])
    rotated_x = QuaternionMath.rotate_vector(q_90z, test_x)
    expected = np.array([0, 1, 0])
    
    print(f"\n90° Z-rotation test:")
    print(f"  Original: {test_x}")
    print(f"  Rotated:  {rotated_x}")
    print(f"  Expected: {expected}")
    print(f"  Error:    {np.linalg.norm(expected - rotated_x):.10f}")
    
    # Test quaternion normalization
    q_test = np.array([0.5, 0.5, 0.5, 0.5])
    q_norm = np.linalg.norm(q_test)
    print(f"\nQuaternion normalization test:")
    print(f"  Before: {q_test}, norm = {q_norm:.6f}")
    q_test_normalized = q_test / q_norm
    print(f"  After:  {q_test_normalized}, norm = {np.linalg.norm(q_test_normalized):.10f}")


def test_control_system():
    """Test PID control system"""
    print("\n=== Testing Control System ===")
    
    config = TVCConfiguration()
    sim = AthenaTVCSimulation(config)
    
    # Step response test
    desired = np.array([0.1, 0.05])  # 0.1 rad pitch, 0.05 rad yaw
    
    time_data = []
    gimbal_data = []
    error_data = []
    
    dt = 0.001
    for i in range(2000):  # 2 seconds
        current = np.array(sim.gimbal.compute_gimbal_attitude())
        control = sim.controller.compute_control(desired, current, dt)
        
        sim.gimbal.update_actuators(dt, control[0], control[1])
        
        time_data.append(i * dt)
        gimbal_data.append(current.copy())
        error_data.append(np.linalg.norm(desired - current))
    
    # Analyze performance
    final_error = error_data[-1]
    max_error = max(error_data)
    settling_time = None
    
    tolerance = 0.01  # 1% tolerance
    for i in range(len(error_data) - 1, -1, -1):
        if error_data[i] > tolerance * np.linalg.norm(desired):
            settling_time = time_data[i + 1] if i + 1 < len(time_data) else time_data[-1]
            break
    
    print(f"Control System Performance:")
    print(f"  Target: {desired}")
    print(f"  Final:  {gimbal_data[-1]}")
    print(f"  Final error: {final_error:.6f} rad")
    print(f"  Max error:   {max_error:.6f} rad")
    print(f"  Settling time: {settling_time:.3f}s")


def test_different_configurations():
    """Test different actuator configurations"""
    print("\n=== Testing Different Configurations ===")
    
    configurations = [
        ("3-actuator (120°)", np.array([0, 2*np.pi/3, 4*np.pi/3])),
        ("4-actuator (90°)", np.array([0, np.pi/2, np.pi, 3*np.pi/2])),
        ("3-actuator (non-uniform)", np.array([0, np.pi/2, np.pi])),
    ]
    
    test_gimbal = (0.1, 0.05)
    
    for name, angles in configurations:
        config = TVCConfiguration()
        config.actuator_angles = angles
        sim = AthenaTVCSimulation(config)
        
        # Compute inverse kinematics
        lengths = sim.gimbal.inverse_kinematics(test_gimbal[0], test_gimbal[1])
        
        # Verify forward kinematics
        for actuator, length in zip(sim.gimbal.actuators, lengths):
            actuator.current_length = length
        
        computed = sim.gimbal.compute_gimbal_attitude()
        error = np.linalg.norm(np.array(test_gimbal) - np.array(computed))
        
        print(f"{name}:")
        print(f"  Actuator angles: {np.degrees(angles)}")
        print(f"  Target gimbal:   {test_gimbal}")
        print(f"  Computed gimbal: {computed}")
        print(f"  Error:           {error:.6f} rad")
        print(f"  Actuator lengths: {lengths}")
        print()


def run_performance_test():
    """Test simulation performance"""
    print("\n=== Performance Test ===")
    
    config = TVCConfiguration()
    sim = AthenaTVCSimulation(config)
    
    # Time simulation steps
    import time
    
    n_steps = 10000
    start_time = time.time()
    
    for i in range(n_steps):
        sim.step_simulation()
    
    end_time = time.time()
    total_time = end_time - start_time
    step_time = total_time / n_steps
    
    print(f"Performance Metrics:")
    print(f"  Steps simulated: {n_steps}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Time per step: {step_time*1000:.3f}ms")
    print(f"  Simulation rate: {1/step_time:.0f} Hz")
    print(f"  Real-time factor: {sim.dt/step_time:.1f}x")


if __name__ == "__main__":
    print("Athena TVC Simulation Test Suite")
    print("================================")
    
    try:
        test_inverse_kinematics()
        test_actuator_dynamics()
        test_quaternion_math()
        test_control_system()
        test_different_configurations()
        run_performance_test()
        
        print("\n=== All Tests Completed Successfully ===")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 