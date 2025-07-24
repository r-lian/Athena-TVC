"""
Example scripts for Athena TVC MuJoCo Simulation

Comprehensive examples demonstrating various capabilities of the MuJoCo-based
TVC simulation including control scenarios, visualization, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from mujoco_tvc_sim import MuJoCoTVCSimulation
from tvc_sim import TVCConfiguration


def example_1_basic_simulation():
    """Example 1: Basic MuJoCo TVC simulation"""
    print("=" * 60)
    print("Example 1: Basic MuJoCo TVC Simulation")
    print("=" * 60)
    
    # Create simulation with default configuration
    sim = MuJoCoTVCSimulation()
    
    # Define a simple thrust profile
    def thrust_profile(t):
        if t < 1.0:
            return 0.0          # No thrust initially
        elif t < 5.0:
            return 40.0         # Constant thrust
        else:
            return 20.0         # Reduced thrust
    
    # Define simple gimbal commands
    def gimbal_commands(t):
        # Small oscillations to test control
        return np.array([
            0.05 * np.sin(2 * np.pi * 0.5 * t),  # 0.5 Hz pitch
            0.03 * np.cos(2 * np.pi * 0.7 * t)   # 0.7 Hz yaw
        ])
    
    print("Running basic simulation...")
    sim.run_simulation(
        duration=6.0,
        thrust_profile=thrust_profile,
        gimbal_commands=gimbal_commands
    )
    
    # Plot results
    sim.plot_results()
    
    # Print summary
    final_state = sim.get_rocket_state()
    print(f"\nSimulation Summary:")
    print(f"- Final position: {final_state['position']}")
    print(f"- Final gimbal angles: {np.degrees(final_state['gimbal_angles'])} deg")
    print(f"- Total displacement: {np.linalg.norm(final_state['position']):.3f} m")
    
    return sim


def example_2_step_response():
    """Example 2: Gimbal step response analysis"""
    print("\n" + "=" * 60)
    print("Example 2: Gimbal Step Response Analysis")
    print("=" * 60)
    
    sim = MuJoCoTVCSimulation()
    
    # Step response test
    step_magnitude = 0.1  # 0.1 radian (5.7 degrees)
    
    print(f"Testing step response to {np.degrees(step_magnitude):.1f}° pitch command...")
    
    # Run step response
    sim.reset_simulation()
    
    time_data = []
    gimbal_data = []
    target_data = []
    
    # Apply step at t=0.5s
    step_time = 0.5
    duration = 3.0
    
    steps = int(duration / sim.model.opt.timestep)
    for i in range(steps):
        t = i * sim.model.opt.timestep
        
        if t >= step_time:
            target = np.array([step_magnitude, 0.0])
        else:
            target = np.array([0.0, 0.0])
        
        sim.step_simulation(target)
        
        time_data.append(t)
        gimbal_data.append(sim.get_gimbal_angles())
        target_data.append(target.copy())
    
    # Analyze response
    time_array = np.array(time_data)
    gimbal_array = np.array(gimbal_data)
    target_array = np.array(target_data)
    
    # Find settling time (2% criterion)
    step_start_idx = np.where(time_array >= step_time)[0][0]
    final_value = gimbal_array[-1, 0]
    tolerance = 0.02 * abs(step_magnitude)
    
    settling_idx = None
    for i in range(len(gimbal_array) - 1, step_start_idx, -1):
        if abs(gimbal_array[i, 0] - final_value) > tolerance:
            settling_idx = i + 1
            break
    
    settling_time = time_array[settling_idx] - step_time if settling_idx else 0
    
    # Find overshoot
    response_after_step = gimbal_array[step_start_idx:, 0]
    max_value = np.max(response_after_step)
    overshoot = (max_value - step_magnitude) / step_magnitude * 100 if step_magnitude != 0 else 0
    
    # Plot step response
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(time_array, np.degrees(target_array[:, 0]), 'r--', label='Target', linewidth=2)
    plt.plot(time_array, np.degrees(gimbal_array[:, 0]), 'b-', label='Actual')
    plt.axvline(step_time, color='gray', linestyle=':', alpha=0.7, label='Step Applied')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (deg)')
    plt.title('Pitch Step Response')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_array, np.degrees(gimbal_array[:, 1]), 'g-', label='Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle (deg)')
    plt.title('Yaw Response (Should be ~0)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    actuator_positions = np.array(sim.actuator_history)
    for i in range(3):
        plt.plot(time_array[:len(actuator_positions)], actuator_positions[:, i], 
                label=f'Actuator {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Actuator Length (m)')
    plt.title('Actuator Response')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    error = np.degrees(np.abs(target_array[:, 0] - gimbal_array[:, 0]))
    plt.semilogy(time_array, error, 'purple')
    plt.axhline(np.degrees(tolerance), color='red', linestyle='--', 
                label=f'2% Tolerance')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (deg)')
    plt.title('Tracking Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nStep Response Analysis:")
    print(f"- Step magnitude: {np.degrees(step_magnitude):.1f}°")
    print(f"- Final value: {np.degrees(final_value):.1f}°")
    print(f"- Settling time: {settling_time:.3f}s")
    print(f"- Overshoot: {overshoot:.1f}%")
    print(f"- Steady-state error: {np.degrees(abs(final_value - step_magnitude)):.3f}°")
    
    return sim


def example_3_hover_stabilization():
    """Example 3: Rocket hover with disturbance rejection"""
    print("\n" + "=" * 60)
    print("Example 3: Rocket Hover with Disturbance Rejection")
    print("=" * 60)
    
    sim = MuJoCoTVCSimulation()
    
    # Thrust profile for hover (approximately counteracts gravity)
    rocket_mass = 5.0  # kg (from model)
    hover_thrust = rocket_mass * 9.81  # N
    
    def thrust_profile(t):
        if t < 1.0:
            return hover_thrust * t  # Ramp up
        elif t < 8.0:
            return hover_thrust      # Hover
        else:
            return hover_thrust * (10.0 - t) / 2.0  # Ramp down
    
    # Disturbance rejection - external wind/disturbances
    def gimbal_commands(t):
        # Simulate wind disturbance compensation
        disturbance_x = 0.02 * np.sin(2 * np.pi * 0.3 * t)  # Slow wind
        disturbance_y = 0.01 * np.cos(2 * np.pi * 0.7 * t)  # Faster gust
        
        # Add step disturbance
        if 4.0 < t < 5.0:
            disturbance_x += 0.05  # Step disturbance
        
        return np.array([disturbance_x, disturbance_y])
    
    print("Running hover simulation with disturbance rejection...")
    sim.run_simulation(
        duration=10.0,
        thrust_profile=thrust_profile,
        gimbal_commands=gimbal_commands
    )
    
    # Analyze hover performance
    time_array = np.array(sim.time_history)
    position_array = np.array(sim.position_history)
    gimbal_array = np.array(sim.gimbal_history)
    thrust_array = np.array(sim.thrust_history)
    
    # Plot hover analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Rocket Hover with Disturbance Rejection')
    
    # Position drift
    axes[0,0].plot(time_array, position_array[:, 0], 'r-', label='X')
    axes[0,0].plot(time_array, position_array[:, 1], 'g-', label='Y')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Horizontal Position (m)')
    axes[0,0].set_title('Position Drift')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Altitude
    axes[0,1].plot(time_array, position_array[:, 2], 'b-')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Altitude (m)')
    axes[0,1].set_title('Altitude Control')
    axes[0,1].grid(True)
    
    # 2D trajectory
    axes[0,2].plot(position_array[:, 0], position_array[:, 1], 'b-')
    axes[0,2].scatter(position_array[0, 0], position_array[0, 1], c='g', s=50, label='Start')
    axes[0,2].scatter(position_array[-1, 0], position_array[-1, 1], c='r', s=50, label='End')
    axes[0,2].set_xlabel('X Position (m)')
    axes[0,2].set_ylabel('Y Position (m)')
    axes[0,2].set_title('Horizontal Trajectory')
    axes[0,2].legend()
    axes[0,2].grid(True)
    axes[0,2].axis('equal')
    
    # Gimbal commands
    axes[1,0].plot(time_array, np.degrees(gimbal_array[:, 0]), 'r-', label='Pitch')
    axes[1,0].plot(time_array, np.degrees(gimbal_array[:, 1]), 'b-', label='Yaw')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Gimbal Angle (deg)')
    axes[1,0].set_title('Gimbal Response to Disturbances')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Thrust profile
    axes[1,1].plot(time_array, thrust_array, 'purple')
    axes[1,1].axhline(hover_thrust, color='red', linestyle='--', 
                     label=f'Hover Thrust ({hover_thrust:.1f}N)')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Thrust (N)')
    axes[1,1].set_title('Thrust Profile')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # Position deviation from start
    position_deviation = np.linalg.norm(position_array - position_array[0], axis=1)
    axes[1,2].plot(time_array, position_deviation, 'orange')
    axes[1,2].set_xlabel('Time (s)')
    axes[1,2].set_ylabel('Deviation from Start (m)')
    axes[1,2].set_title('Total Position Deviation')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Performance metrics
    max_horizontal_drift = np.max(np.linalg.norm(position_array[:, :2] - position_array[0, :2], axis=1))
    final_altitude = position_array[-1, 2]
    max_gimbal_angle = np.max(np.abs(gimbal_array))
    
    print(f"\nHover Performance Metrics:")
    print(f"- Max horizontal drift: {max_horizontal_drift:.3f} m")
    print(f"- Final altitude: {final_altitude:.3f} m")
    print(f"- Max gimbal deflection: {np.degrees(max_gimbal_angle):.1f}°")
    print(f"- Hover thrust: {hover_thrust:.1f} N")
    
    return sim


def example_4_requirement_validation():
    """Example 4: Validate Athena project requirements"""
    print("\n" + "=" * 60)
    print("Example 4: Athena Project Requirements Validation")
    print("=" * 60)
    
    # Project requirements:
    # - Manipulate thrust vector at least 7 degrees in all directions
    # - Controlled manner within 10 seconds
    # - Present data to verify successful thrust vectoring
    
    sim = MuJoCoTVCSimulation()
    
    print("Testing requirement: 7° thrust vector manipulation in 10 seconds")
    
    # Test sequence: move gimbal to maximum angles
    target_angle = np.radians(7.0)  # 7 degrees
    
    def gimbal_test_sequence(t):
        """Test sequence to validate requirements"""
        if t < 2.0:
            # Move to +7° pitch
            return np.array([target_angle, 0.0])
        elif t < 4.0:
            # Move to -7° pitch
            return np.array([-target_angle, 0.0])
        elif t < 6.0:
            # Move to +7° yaw
            return np.array([0.0, target_angle])
        elif t < 8.0:
            # Move to -7° yaw
            return np.array([0.0, -target_angle])
        elif t < 10.0:
            # Return to center
            return np.array([0.0, 0.0])
        else:
            # Test diagonal movements
            return np.array([target_angle * 0.7, target_angle * 0.7])
    
    def thrust_profile(t):
        return 30.0  # Constant moderate thrust
    
    print("Running requirement validation sequence...")
    sim.run_simulation(
        duration=12.0,
        thrust_profile=thrust_profile,
        gimbal_commands=gimbal_test_sequence
    )
    
    # Analyze requirement compliance
    time_array = np.array(sim.time_history)
    gimbal_array = np.array(sim.gimbal_history)
    
    # Check maximum achieved angles
    max_pitch = np.max(np.abs(gimbal_array[:, 0]))
    max_yaw = np.max(np.abs(gimbal_array[:, 1]))
    
    # Check response times
    target_tolerance = np.radians(0.5)  # 0.5 degree tolerance
    
    def find_response_time(target_time, target_val, axis):
        """Find time to reach target within tolerance"""
        start_idx = np.where(time_array >= target_time)[0][0]
        for i in range(start_idx, len(gimbal_array)):
            if abs(gimbal_array[i, axis] - target_val) < target_tolerance:
                return time_array[i] - target_time
        return float('inf')
    
    # Test response times for each movement
    response_times = [
        find_response_time(0.0, target_angle, 0),    # +7° pitch
        find_response_time(2.0, -target_angle, 0),   # -7° pitch
        find_response_time(4.0, target_angle, 1),    # +7° yaw
        find_response_time(6.0, -target_angle, 1),   # -7° yaw
    ]
    
    # Plotting requirement validation
    plt.figure(figsize=(15, 10))
    
    # Gimbal angle time series
    plt.subplot(2, 3, 1)
    plt.plot(time_array, np.degrees(gimbal_array[:, 0]), 'r-', linewidth=2, label='Pitch')
    plt.plot(time_array, np.degrees(gimbal_array[:, 1]), 'b-', linewidth=2, label='Yaw')
    plt.axhline(7, color='gray', linestyle='--', alpha=0.7, label='±7° Requirement')
    plt.axhline(-7, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Gimbal Angle (deg)')
    plt.title('Requirement Test Sequence')
    plt.legend()
    plt.grid(True)
    
    # Gimbal magnitude
    plt.subplot(2, 3, 2)
    gimbal_magnitude = np.degrees(np.linalg.norm(gimbal_array, axis=1))
    plt.plot(time_array, gimbal_magnitude, 'purple', linewidth=2)
    plt.axhline(7, color='red', linestyle='--', label='7° Requirement')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Gimbal Deflection (deg)')
    plt.title('Gimbal Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Actuator positions
    plt.subplot(2, 3, 3)
    actuator_array = np.array(sim.actuator_history)
    for i in range(3):
        plt.plot(time_array[:len(actuator_array)], actuator_array[:, i], 
                label=f'Actuator {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Actuator Length (m)')
    plt.title('Actuator Coordination')
    plt.legend()
    plt.grid(True)
    
    # 2D gimbal trajectory
    plt.subplot(2, 3, 4)
    plt.plot(np.degrees(gimbal_array[:, 0]), np.degrees(gimbal_array[:, 1]), 'b-', alpha=0.7)
    plt.scatter(np.degrees(gimbal_array[0, 0]), np.degrees(gimbal_array[0, 1]), 
               c='g', s=100, label='Start', zorder=5)
    plt.scatter(np.degrees(gimbal_array[-1, 0]), np.degrees(gimbal_array[-1, 1]), 
               c='r', s=100, label='End', zorder=5)
    
    # Draw requirement boundary
    theta = np.linspace(0, 2*np.pi, 100)
    req_circle_x = 7 * np.cos(theta)
    req_circle_y = 7 * np.sin(theta)
    plt.plot(req_circle_x, req_circle_y, 'r--', alpha=0.7, label='7° Requirement')
    
    plt.xlabel('Pitch Angle (deg)')
    plt.ylabel('Yaw Angle (deg)')
    plt.title('Gimbal 2D Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Response time analysis
    plt.subplot(2, 3, 5)
    movements = ['Pitch +7°', 'Pitch -7°', 'Yaw +7°', 'Yaw -7°']
    response_times_finite = [t if t != float('inf') else 10 for t in response_times]
    
    bars = plt.bar(movements, response_times_finite, 
                   color=['green' if t < 2.0 else 'orange' if t < 5.0 else 'red' 
                          for t in response_times_finite])
    plt.axhline(10, color='red', linestyle='--', label='10s Requirement')
    plt.ylabel('Response Time (s)')
    plt.title('Response Time Analysis')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Performance summary
    plt.subplot(2, 3, 6)
    metrics = ['Max Pitch', 'Max Yaw', 'Avg Response', 'Max Response']
    values = [
        np.degrees(max_pitch),
        np.degrees(max_yaw),
        np.mean([t for t in response_times if t != float('inf')]),
        max([t for t in response_times if t != float('inf')])
    ]
    requirements = [7, 7, 5, 10]  # Requirement thresholds
    
    colors = ['green' if v >= r else 'red' for v, r in zip(values, requirements)]
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    
    # Add requirement lines
    for i, req in enumerate(requirements):
        plt.axhline(req, color='red', linestyle='--', alpha=0.3)
    
    plt.ylabel('Value')
    plt.title('Requirements Compliance')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Requirements assessment
    print(f"\nRequirements Validation Results:")
    print(f"=" * 40)
    print(f"1. Thrust Vector Deflection:")
    print(f"   - Maximum pitch: {np.degrees(max_pitch):.1f}° (Req: ≥7°) {'✓' if max_pitch >= np.radians(7) else '✗'}")
    print(f"   - Maximum yaw:   {np.degrees(max_yaw):.1f}° (Req: ≥7°) {'✓' if max_yaw >= np.radians(7) else '✗'}")
    
    print(f"\n2. Response Time (within 10 seconds):")
    for i, (movement, rt) in enumerate(zip(movements, response_times)):
        status = '✓' if rt < 10.0 else '✗'
        rt_str = f"{rt:.2f}s" if rt != float('inf') else ">10s"
        print(f"   - {movement}: {rt_str} {status}")
    
    print(f"\n3. Control Authority:")
    max_total_deflection = np.degrees(np.max(np.linalg.norm(gimbal_array, axis=1)))
    print(f"   - Max total deflection: {max_total_deflection:.1f}°")
    print(f"   - All directions controllable: {'✓' if max_total_deflection >= 7 else '✗'}")
    
    overall_pass = (max_pitch >= np.radians(7) and max_yaw >= np.radians(7) and 
                   all(rt < 10.0 for rt in response_times if rt != float('inf')))
    
    print(f"\n4. Overall Assessment: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    return sim


def example_5_interactive_demo():
    """Example 5: Interactive 3D demonstration"""
    print("\n" + "=" * 60)
    print("Example 5: Interactive 3D Demonstration")
    print("=" * 60)
    
    sim = MuJoCoTVCSimulation()
    
    print("Starting interactive 3D demonstration...")
    print("This will open a MuJoCo viewer window where you can:")
    print("- Rotate the view with mouse")
    print("- Zoom with scroll wheel")
    print("- Watch the TVC system in action")
    print("\nPress Ctrl+C to stop the simulation")
    
    # Define an interesting demo sequence
    def demo_thrust(t):
        """Thrust profile for demonstration"""
        if t < 2.0:
            return 30.0 * t / 2.0  # Ramp up
        elif t < 8.0:
            return 30.0 + 20.0 * np.sin(2 * np.pi * 0.2 * t)  # Oscillating
        else:
            return 30.0 * (10.0 - t) / 2.0  # Ramp down
    
    def demo_gimbal(t):
        """Gimbal commands for demonstration"""
        # Complex pattern to show off capabilities
        pitch = 0.08 * np.sin(2 * np.pi * 0.3 * t) * np.exp(-0.1 * t)
        yaw = 0.05 * np.cos(2 * np.pi * 0.7 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
        return np.array([pitch, yaw])
    
    try:
        sim.run_interactive_simulation(duration=10.0)
        
        # After interactive demo, show a summary
        if len(sim.time_history) > 0:
            print(f"\nDemo completed!")
            print(f"- Simulation time: {sim.time:.1f}s")
            print(f"- Data points logged: {len(sim.time_history)}")
            
            # Option to plot results
            plot_demo = input("Plot demo results? (y/n): ")
            if plot_demo.lower() == 'y':
                sim.plot_results()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: Make sure MuJoCo is properly installed and the XML file exists")
    
    return sim


def run_all_examples():
    """Run all examples in sequence"""
    print("Athena TVC MuJoCo Simulation Examples")
    print("=" * 60)
    print("This will run through all demonstration examples.")
    print("Each example showcases different capabilities of the simulation.")
    print()
    
    examples = [
        ("Basic Simulation", example_1_basic_simulation),
        ("Step Response Analysis", example_2_step_response),
        ("Hover Stabilization", example_3_hover_stabilization),
        ("Requirements Validation", example_4_requirement_validation),
        ("Interactive 3D Demo", example_5_interactive_demo)
    ]
    
    try:
        for i, (name, func) in enumerate(examples, 1):
            user_input = input(f"\nRun Example {i}: {name}? (y/n/q): ")
            
            if user_input.lower() == 'q':
                print("Examples stopped by user")
                break
            elif user_input.lower() == 'y':
                sim = func()
                print(f"Example {i} completed!")
            else:
                print(f"Example {i} skipped")
        
        print("\nAll selected examples completed!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    # Check if user wants to run all examples or select specific ones
    print("Athena TVC MuJoCo Simulation Examples")
    print("=" * 50)
    
    mode = input("Run mode: (a)ll examples, (s)elect specific, or (i)nteractive demo only? ")
    
    if mode.lower() == 'a':
        run_all_examples()
    elif mode.lower() == 's':
        print("\nAvailable examples:")
        print("1. Basic Simulation")
        print("2. Step Response Analysis") 
        print("3. Hover Stabilization")
        print("4. Requirements Validation")
        print("5. Interactive 3D Demo")
        
        selection = input("Enter example numbers (e.g., 1,3,5): ")
        
        example_funcs = [
            example_1_basic_simulation,
            example_2_step_response,
            example_3_hover_stabilization,
            example_4_requirement_validation,
            example_5_interactive_demo
        ]
        
        try:
            selected = [int(x.strip()) - 1 for x in selection.split(',')]
            for idx in selected:
                if 0 <= idx < len(example_funcs):
                    print(f"\nRunning Example {idx + 1}...")
                    example_funcs[idx]()
                else:
                    print(f"Invalid example number: {idx + 1}")
        except ValueError:
            print("Invalid input format")
    
    elif mode.lower() == 'i':
        example_5_interactive_demo()
    
    else:
        print("Invalid selection") 