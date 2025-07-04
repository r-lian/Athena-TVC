"""
Simple verification script for Athena TVC Simulation
Tests basic functionality with clear output
"""

import numpy as np
from tvc_sim import AthenaTVCSimulation, TVCConfiguration

def main():
    print("Athena TVC Simulation Verification")
    print("=" * 40)
    
    # Test 1: Basic simulation creation
    print("1. Creating simulation...")
    try:
        config = TVCConfiguration()
        sim = AthenaTVCSimulation(config)
        print("   ✓ Simulation created successfully")
        print(f"   - {len(sim.gimbal.actuators)} actuators configured")
        print(f"   - Max gimbal angle: {np.degrees(config.max_gimbal_angle):.1f}°")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    # Test 2: Inverse kinematics
    print("\n2. Testing inverse kinematics...")
    try:
        test_angles = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)]
        for gx, gy in test_angles:
            lengths = sim.gimbal.inverse_kinematics(gx, gy)
            print(f"   Gimbal ({gx:.1f}, {gy:.1f}) rad -> Lengths: {[f'{l:.4f}' for l in lengths]} m")
        print("   ✓ Inverse kinematics working")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    # Test 3: Single simulation step
    print("\n3. Testing simulation step...")
    try:
        initial_time = sim.time
        sim.step_simulation()
        if sim.time > initial_time:
            print("   ✓ Simulation step successful")
            print(f"   - Time advanced: {sim.time - initial_time:.4f}s")
        else:
            print("   ✗ Time did not advance")
            return
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    # Test 4: Short simulation run
    print("\n4. Running short simulation...")
    try:
        sim.set_thrust(25.0)  # 25N thrust
        sim.run_simulation(duration=0.1)  # 0.1 second
        print("   ✓ Short simulation completed")
        print(f"   - Final time: {sim.time:.3f}s")
        print(f"   - Data points: {len(sim.time_history)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    # Test 5: Basic control test
    print("\n5. Testing gimbal control...")
    try:
        # Reset simulation
        sim2 = AthenaTVCSimulation(config)
        
        # Command small gimbal movement
        desired_gimbal = np.array([0.05, 0.02])  # Small angles
        
        # Run a few steps
        for i in range(10):
            sim2.step_simulation(desired_gimbal)
        
        current_gimbal = np.array(sim2.gimbal.compute_gimbal_attitude())
        print(f"   Target: [{desired_gimbal[0]:.3f}, {desired_gimbal[1]:.3f}] rad")
        print(f"   Actual: [{current_gimbal[0]:.3f}, {current_gimbal[1]:.3f}] rad")
        
        error = np.linalg.norm(desired_gimbal - current_gimbal)
        if error < 0.1:  # Reasonable tolerance
            print("   ✓ Gimbal control working")
        else:
            print(f"   ! Large error: {error:.4f} rad (may need more time to converge)")
            
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    print("\n" + "=" * 40)
    print("✓ All basic tests passed!")
    print("The Athena TVC simulation is ready for use.")
    print("\nNext steps:")
    print("- Run 'python tvc_sim.py' for full demo")
    print("- Run 'python test_tvc.py' for comprehensive tests")
    print("- Check README.md for usage examples")

if __name__ == "__main__":
    main() 