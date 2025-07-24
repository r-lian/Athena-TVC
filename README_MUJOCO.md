# Athena TVC MuJoCo Simulation

A physics-accurate thrust vector control (TVC) simulation using MuJoCo for the Athena rocket project. This simulation provides realistic multi-body dynamics, 3D visualization, and comprehensive analysis tools for TVC system development.

## 🚀 Project Overview

The Athena TVC project aims to develop a thrust vector controlled system to inform future TVC designs. This MuJoCo simulation provides:

- **Realistic Physics**: Multi-body dynamics with proper constraints and forces
- **3D Visualization**: Real-time interactive visualization of the TVC system
- **Control Validation**: Test control algorithms and validate requirements
- **Hardware Integration**: Bridge between simulation and real hardware

### Project Requirements Validation

✅ **Manipulate thrust vector ≥7° in all directions**  
✅ **Controlled manner within 10 seconds**  
✅ **Present data to verify successful thrust vectoring**

## 📋 Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux
- OpenGL support for visualization

## 🛠️ Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
python setup_mujoco.py
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python setup_mujoco.py --check-only
```

### Dependencies

- `mujoco>=2.3.0` - Physics simulation engine
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Plotting and visualization
- `imageio>=2.9.0` - Image/video processing
- `opencv-python>=4.5.0` - Computer vision utilities

## 🚁 Quick Start

### 1. Basic Simulation

```python
from mujoco_tvc_sim import MuJoCoTVCSimulation

# Create simulation
sim = MuJoCoTVCSimulation()

# Run basic simulation
sim.run_simulation(duration=5.0)
sim.plot_results()
```

### 2. Interactive 3D Visualization

```python
# Start interactive simulation with 3D viewer
sim.run_interactive_simulation(duration=10.0)
```

### 3. Custom Control Scenarios

```python
# Define thrust profile
def thrust_profile(t):
    return 50.0 if t < 5.0 else 20.0

# Define gimbal commands
def gimbal_commands(t):
    return np.array([0.1 * np.sin(t), 0.05 * np.cos(t)])

# Run simulation
sim.run_simulation(
    duration=8.0,
    thrust_profile=thrust_profile,
    gimbal_commands=gimbal_commands
)
```

## 📊 Examples and Testing

### Run Comprehensive Tests

```bash
# Test all functionality
python test_mujoco_tvc.py

# Compare with original simulation
python test_mujoco_tvc.py --compare
```

### Explore Examples

```bash
# Run all demonstration examples
python examples_mujoco_tvc.py

# Run specific examples
python examples_mujoco_tvc.py --select 1,3,5
```

### Available Examples

1. **Basic Simulation** - Simple TVC demonstration
2. **Step Response Analysis** - Control system characterization
3. **Hover Stabilization** - Disturbance rejection testing
4. **Requirements Validation** - Verify Athena project requirements
5. **Interactive 3D Demo** - Real-time visualization

## 🎮 Interactive Controls

### MuJoCo Viewer Controls

- **Mouse**: Rotate view
- **Scroll**: Zoom in/out
- **Shift+Mouse**: Pan view
- **Ctrl+C**: Stop simulation
- **Space**: Pause/resume

### Real-time Control Interface

```python
# Custom control loop
while running:
    # Get current state
    state = sim.get_rocket_state()
    
    # Compute control commands
    gimbal_cmd = your_controller(state)
    
    # Step simulation
    sim.step_simulation(gimbal_cmd)
```

## 🔧 Configuration

### TVC System Parameters

```python
from tvc_sim import TVCConfiguration

config = TVCConfiguration(
    gimbal_radius=0.05,           # 5cm gimbal radius
    actuator_min_length=0.08,     # 8cm minimum actuator length
    actuator_max_length=0.12,     # 12cm maximum actuator length
    max_gimbal_angle=0.15,        # ±8.6° maximum gimbal deflection
    max_thrust=100.0              # 100N maximum thrust
)

sim = MuJoCoTVCSimulation(config=config)
```

### Model Customization

Edit `athena_tvc.xml` to modify:
- Geometry and mass properties
- Actuator characteristics
- Sensor configurations
- Visual appearance

## 📈 Analysis Tools

### Performance Metrics

```python
# Get simulation results
results = {
    'time': sim.time_history,
    'gimbal_angles': sim.gimbal_history,
    'actuator_positions': sim.actuator_history,
    'rocket_state': sim.get_rocket_state()
}

# Analyze response characteristics
sim.plot_results()
```

### Comparison with Original Simulation

```python
from tvc_sim import AthenaTVCSimulation

original_sim = AthenaTVCSimulation()
sim.compare_with_original(original_sim, duration=5.0)
```

### Export Data

```python
import numpy as np

# Export for external analysis
np.save('simulation_data.npy', {
    'time': sim.time_history,
    'gimbal': sim.gimbal_history,
    'position': sim.position_history
})
```

## 🔬 Technical Details

### Model Architecture

- **Rocket Body**: Main structure with realistic inertia
- **Gimbal System**: Two-axis gimbal with pitch/yaw joints
- **Linear Actuators**: Three actuators at 120° spacing
- **Engine**: Thrust application point with nozzle geometry
- **Constraints**: Ball joint connections between actuators and gimbal

### Physics Simulation

- **Integrator**: Runge-Kutta 4th order (RK4)
- **Timestep**: 1ms (configurable)
- **Gravity**: 9.81 m/s² (Earth gravity)
- **Friction**: Realistic joint friction and damping

### Control Interface

```python
# Low-level actuator control
sim.set_actuator_targets([0.09, 0.10, 0.095])  # meters

# High-level gimbal control
desired_angles = np.array([0.1, 0.05])  # pitch, yaw in radians
lengths = sim.inverse_kinematics(desired_angles[0], desired_angles[1])
sim.set_actuator_targets(lengths)

# Apply thrust
sim.apply_thrust(50.0)  # Newtons
```

## 🐛 Troubleshooting

### Common Issues

**MuJoCo Import Error**
```bash
pip install mujoco>=2.3.0
# or
conda install -c conda-forge mujoco
```

**OpenGL/Viewer Issues**
- Update graphics drivers
- Check OpenGL support
- Try software rendering: `export MUJOCO_GL=osmesa`

**Model Loading Errors**
- Verify `athena_tvc.xml` exists in current directory
- Check XML syntax
- Run `python setup_mujoco.py --check-only`

**Performance Issues**
- Reduce simulation duration
- Increase timestep (trade accuracy for speed)
- Disable visualization for batch runs

### Getting Help

1. Check the troubleshooting section above
2. Run diagnostics: `python setup_mujoco.py --check-only`
3. Review MuJoCo documentation: https://mujoco.readthedocs.io/
4. Check existing simulation: `python verify_sim.py`

## 🚀 Advanced Usage

### Hardware-in-the-Loop (HIL)

```python
class HILInterface:
    def __init__(self, sim):
        self.sim = sim
        self.hardware = YourHardwareInterface()
    
    def run_hil_test(self):
        while True:
            # Read from hardware
            sensor_data = self.hardware.read_sensors()
            
            # Update simulation
            self.sim.step_simulation()
            
            # Send to hardware
            actuator_commands = self.sim.get_actuator_positions()
            self.hardware.set_actuators(actuator_commands)
```

### Batch Simulation

```python
# Parameter sweep
results = []
for thrust in [30, 50, 70]:
    for angle in [0.05, 0.10, 0.15]:
        sim.reset_simulation()
        sim.run_simulation(duration=5.0, thrust_profile=lambda t: thrust)
        results.append(sim.get_rocket_state())
```

### Custom Visualizations

```python
# Real-time plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_simulation(sim):
    fig, ax = plt.subplots()
    
    def update(frame):
        sim.step_simulation()
        state = sim.get_rocket_state()
        # Update plot with new state
        
    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
```

## 📝 File Structure

```
Rocket Team TVC Project/
├── README.md                 # Original project README
├── README_MUJOCO.md         # This file
├── requirements.txt         # Python dependencies
├── athena_tvc.xml          # MuJoCo model definition
├── mujoco_tvc_sim.py       # Main MuJoCo simulation
├── test_mujoco_tvc.py      # Comprehensive tests
├── examples_mujoco_tvc.py  # Example demonstrations
├── setup_mujoco.py         # Setup and verification
├── tvc_sim.py              # Original simulation
├── test_tvc.py             # Original tests
└── verify_sim.py           # Basic verification
```

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Test with both simulations

## 📄 License

This project is designed for educational and research purposes in rocket propulsion and control systems.

## 🙏 Acknowledgments

- MuJoCo physics engine by DeepMind
- Original TVC simulation framework
- Athena rocket project team

---

**Ready to explore thrust vector control? Start with the interactive demo:**

```bash
python examples_mujoco_tvc.py --interactive
``` 