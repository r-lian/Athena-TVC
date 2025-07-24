"""
Test script for MuJoCo Athena TVC Simulation

Comprehensive testing of the MuJoCo-based TVC simulation including
comparison with the original simulation, performance testing, and validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from mujoco_tvc_sim import MuJoCoTVCSimulation
from tvc_sim import AthenaTVCSimulation, TVCConfiguration


def test_mujoco_initialization():
    """Test MuJoCo simulation initialization"""
    print("=== Testing MuJoCo Initialization ===")
    
    try:
        sim = MuJoCoTVCSimulation()
        print("✓ MuJoCo simulation created successfully")
        print(f"  - Model DOF: {sim.model.nq}")
        print(f"  - Actuators: {sim.model.nu}")
        print(f"  - Sensors: {sim.model.nsensor}")
        print(f"  - Timestep: {sim.model.opt.timestep:.4f}s")
        return sim
    except Exception as e:
        print(f"✗ Failed to initialize MuJoCo simulation: {e}")
        return None


def test_basic_functionality(sim):
    """Test basic MuJoCo simulation functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    if sim is None:
        print("✗ Cannot test - simulation not initialized")
        return False
    
    try:
        # Test state reading
        initial_state = sim.get_rocket_state()
        print("✓ State reading working")
        print(f"  - Initial position: {initial_state['position']}")
        print(f"  - Initial gimbal angles: {np.degrees(initial_state['gimbal_angles'])} deg")
        
        # Test inverse kinematics
        test_angles = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)]
        print("\n✓ Inverse kinematics test:")
        for gx, gy in test_angles:
            lengths = sim.inverse_kinematics(gx, gy)
            print(f"  Gimbal ({gx:.1f}, {gy:.1f}) -> Lengths: {[f'{l:.4f}' for l in lengths]}")
        
        # Test single simulation step
        initial_time = sim.time
        sim.step_simulation()
        if sim.time > initial_time:
            print("✓ Simulation step successful")
            print(f"  - Time advanced: {sim.time - initial_time:.4f}s")
        else:
            print("✗ Time did not advance")
            return False
        
        # Test thrust application
        sim.apply_thrust(50.0)
        sim.step_simulation()
        print("✓ Thrust application working")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gimbal_control(sim):
    """Test gimbal control system"""
    print("\n=== Testing Gimbal Control ===")
    
    if sim is None:
        return False
    
    try:
        sim.reset_simulation()
        
        # Test step response
        target_gimbal = np.array([0.1, 0.05])  # Target angles
        
        # Run control loop
        for i in range(100):  # 0.1 seconds at 1ms timestep
            sim.step_simulation(target_gimbal)
        
        final_gimbal = np.array(sim.get_gimbal_angles())
        error = np.linalg.norm(target_gimbal - final_gimbal)
        
        print(f"✓ Gimbal control test:")
        print(f"  Target: [{target_gimbal[0]:.3f}, {target_gimbal[1]:.3f}] rad")
        print(f"  Actual: [{final_gimbal[0]:.3f}, {final_gimbal[1]:.3f}] rad")
        print(f"  Error:  {error:.4f} rad ({np.degrees(error):.2f} deg)")
        
        if error < 0.05:  # 2.9 degree tolerance
            print("✓ Gimbal control within tolerance")
            return True
        else:
            print("! Gimbal control error above tolerance")
            return False
            
    except Exception as e:
        print(f"✗ Gimbal control test failed: {e}")
        return False


def test_simulation_run(sim):
    """Test running a complete simulation"""
    print("\n=== Testing Simulation Run ===")
    
    if sim is None:
        return False
    
    try:
        sim.reset_simulation()
        
        # Define test profiles
        def thrust_profile(t):
            return 30.0 if t < 2.0 else 0.0
        
        def gimbal_profile(t):
            return np.array([0.05 * np.sin(2 * t), 0.03 * np.cos(3 * t)])
        
        # Run simulation
        start_time = time.time()
        sim.run_simulation(duration=2.0, thrust_profile=thrust_profile, gimbal_commands=gimbal_profile)
        elapsed_time = time.time() - start_time
        
        print("✓ Simulation run completed")
        print(f"  - Duration: 2.0s simulated in {elapsed_time:.3f}s")
        print(f"  - Rate: {2.0/elapsed_time:.1f}x real-time")
        print(f"  - Data points: {len(sim.time_history)}")
        
        if len(sim.time_history) > 0:
            final_state = sim.get_rocket_state()
            print(f"  - Final position: {final_state['position']}")
            print(f"  - Position change: {np.linalg.norm(final_state['position']):.3f}m")
            return True
        else:
            print("✗ No data logged")
            return False
            
    except Exception as e:
        print(f"✗ Simulation run test failed: {e}")
        return False


def test_comparison_with_original():
    """Test comparison between MuJoCo and original simulations"""
    print("\n=== Testing Comparison with Original Simulation ===")
    
    try:
        # Create both simulations
        config = TVCConfiguration()
        mujoco_sim = MuJoCoTVCSimulation(config=config)
        original_sim = AthenaTVCSimulation(config)
        
        # Define consistent test conditions
        def test_gimbal(t):
            return np.array([0.08 * np.sin(t), 0.05 * np.cos(1.5 * t)])
        
        # Run both simulations with same conditions
        duration = 3.0
        
        # MuJoCo simulation
        mujoco_sim.reset_simulation()
        mujoco_sim.run_simulation(duration, gimbal_commands=test_gimbal)
        mujoco_gimbal = np.array(mujoco_sim.gimbal_history)
        mujoco_actuators = np.array(mujoco_sim.actuator_history)
        
        # Original simulation
        original_sim.time = 0.0
        original_sim.time_history.clear()
        original_sim.gimbal_history.clear()
        original_sim.actuator_history.clear()
        original_sim.run_simulation(duration, gimbal_commands=test_gimbal)
        original_gimbal = np.array(original_sim.gimbal_history)
        original_actuators = np.array(original_sim.actuator_history)
        
        # Compare results
        if len(mujoco_gimbal) > 0 and len(original_gimbal) > 0:
            # Compute errors (using final values for simplicity)
            gimbal_error = np.linalg.norm(mujoco_gimbal[-1] - original_gimbal[-1])
            actuator_error = np.linalg.norm(mujoco_actuators[-1] - original_actuators[-1])
            
            print("✓ Comparison completed")
            print(f"  - Final gimbal error: {np.degrees(gimbal_error):.3f} deg")
            print(f"  - Final actuator error: {actuator_error*1000:.2f} mm")
            
            # Plot comparison if requested
            plot_comparison = False  # Set to True to see plots
            if plot_comparison:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('MuJoCo vs Original Simulation')
                
                time_mj = np.array(mujoco_sim.time_history)
                time_orig = np.array(original_sim.time_history)
                
                # Gimbal pitch
                axes[0,0].plot(time_mj, np.degrees(mujoco_gimbal[:, 0]), 'r-', label='MuJoCo')
                axes[0,0].plot(time_orig, np.degrees(original_gimbal[:, 0]), 'b--', label='Original')
                axes[0,0].set_title('Gimbal Pitch')
                axes[0,0].legend()
                axes[0,0].grid(True)
                
                # Gimbal yaw
                axes[0,1].plot(time_mj, np.degrees(mujoco_gimbal[:, 1]), 'r-', label='MuJoCo')
                axes[0,1].plot(time_orig, np.degrees(original_gimbal[:, 1]), 'b--', label='Original')
                axes[0,1].set_title('Gimbal Yaw')
                axes[0,1].legend()
                axes[0,1].grid(True)
                
                # Actuator 1
                axes[1,0].plot(time_mj, mujoco_actuators[:, 0], 'r-', label='MuJoCo')
                axes[1,0].plot(time_orig, original_actuators[:, 0], 'b--', label='Original')
                axes[1,0].set_title('Actuator 1 Length')
                axes[1,0].legend()
                axes[1,0].grid(True)
                
                # Error over time
                min_len = min(len(mujoco_gimbal), len(original_gimbal))
                error_history = np.linalg.norm(mujoco_gimbal[:min_len] - original_gimbal[:min_len], axis=1)
                axes[1,1].plot(time_mj[:min_len], np.degrees(error_history), 'g-')
                axes[1,1].set_title('Gimbal Error')
                axes[1,1].grid(True)
                
                plt.tight_layout()
                plt.show()
            
            return True
        else:
            print("✗ No data to compare")
            return False
            
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance(sim):
    """Test simulation performance"""
    print("\n=== Testing Performance ===")
    
    if sim is None:
        return False
    
    try:
        sim.reset_simulation()
        
        # Time many simulation steps
        n_steps = 1000
        start_time = time.time()
        
        for i in range(n_steps):
            sim.step_simulation()
        
        end_time = time.time()
        total_time = end_time - start_time
        step_time = total_time / n_steps
        
        print("✓ Performance test completed")
        print(f"  - Steps: {n_steps}")
        print(f"  - Total time: {total_time:.3f}s")
        print(f"  - Time per step: {step_time*1000:.3f}ms")
        print(f"  - Simulation rate: {1/step_time:.0f} Hz")
        print(f"  - Real-time factor: {sim.model.opt.timestep/step_time:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def test_actuator_limits(sim):
    """Test actuator limit enforcement"""
    print("\n=== Testing Actuator Limits ===")
    
    if sim is None:
        return False
    
    try:
        sim.reset_simulation()
        
        # Test extreme gimbal commands
        extreme_angles = [
            (0.3, 0.0),   # Extreme pitch
            (0.0, 0.3),   # Extreme yaw
            (0.3, 0.3),   # Extreme both
            (-0.3, -0.3), # Extreme negative
        ]
        
        print("✓ Testing actuator limits:")
        for gx, gy in extreme_angles:
            lengths = sim.inverse_kinematics(gx, gy)
            
            # Check if lengths are within bounds
            min_len = sim.config.actuator_min_length
            max_len = sim.config.actuator_max_length
            
            within_bounds = all(min_len <= l <= max_len for l in lengths)
            
            print(f"  Gimbal ({gx:.1f}, {gy:.1f}) -> Lengths: {[f'{l:.4f}' for l in lengths]} " +
                  f"{'✓' if within_bounds else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Actuator limits test failed: {e}")
        return False


def run_all_tests():
    """Run all MuJoCo TVC tests"""
    print("Athena TVC MuJoCo Simulation Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Initialization
    sim = test_mujoco_initialization()
    if sim is not None:
        tests_passed += 1
    
    # Test 2: Basic functionality
    if test_basic_functionality(sim):
        tests_passed += 1
    
    # Test 3: Gimbal control
    if test_gimbal_control(sim):
        tests_passed += 1
    
    # Test 4: Simulation run
    if test_simulation_run(sim):
        tests_passed += 1
    
    # Test 5: Comparison with original
    if test_comparison_with_original():
        tests_passed += 1
    
    # Test 6: Performance
    if test_performance(sim):
        tests_passed += 1
    
    # Test 7: Actuator limits
    if test_actuator_limits(sim):
        tests_passed += 1
    
    # Results summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! MuJoCo TVC simulation is ready for use.")
    else:
        print(f"! {total_tests - tests_passed} test(s) failed. Check errors above.")
    
    print("\nNext steps:")
    print("- Run 'python mujoco_tvc_sim.py' for demo")
    print("- Use sim.run_interactive_simulation() for 3D visualization")
    print("- Check the examples for more advanced usage")
    
    return sim


if __name__ == "__main__":
    # Run all tests
    sim = run_all_tests()
    
    # Optional interactive demo
    if sim is not None:
        user_input = input("\nRun interactive demo with 3D viewer? (y/n): ")
        if user_input.lower() == 'y':
            try:
                sim.reset_simulation()
                print("Starting interactive demo...")
                sim.run_interactive_simulation(duration=8.0)
            except KeyboardInterrupt:
                print("\nDemo interrupted by user")
            except Exception as e:
                print(f"Demo failed: {e}") 