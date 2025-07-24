"""
Setup script for Athena TVC MuJoCo Simulation

This script helps set up the MuJoCo environment for the TVC simulation,
installs required dependencies, and verifies the installation.
"""

import subprocess
import sys
import os
import importlib.util


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"✅ {package_name} is installed")
            return True
        else:
            print(f"❌ {package_name} is not installed")
            return False
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False


def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                      check=True, capture_output=True)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}")
        print(f"   Error: {e}")
        return False


def install_requirements():
    """Install all required packages"""
    print("\n" + "="*50)
    print("Installing Required Packages")
    print("="*50)
    
    # Required packages with their import names
    requirements = [
        ("numpy>=1.21.0", "numpy"),
        ("matplotlib>=3.5.0", "matplotlib"),
        ("mujoco>=2.3.0", "mujoco"),
        ("imageio>=2.9.0", "imageio"),
        ("opencv-python>=4.5.0", "cv2")
    ]
    
    all_installed = True
    
    for package_spec, import_name in requirements:
        package_name = package_spec.split(">=")[0]
        if not check_package(package_name, import_name):
            if not install_package(package_spec):
                all_installed = False
    
    return all_installed


def verify_mujoco_installation():
    """Verify MuJoCo installation and basic functionality"""
    print("\n" + "="*50)
    print("Verifying MuJoCo Installation")
    print("="*50)
    
    try:
        import mujoco
        print("✅ MuJoCo import successful")
        
        # Check MuJoCo version
        print(f"✅ MuJoCo version: {mujoco.__version__}")
        
        # Test basic MuJoCo functionality
        print("Testing basic MuJoCo functionality...")
        
        # Create a simple model to test
        simple_xml = """
        <mujoco>
            <worldbody>
                <geom name="floor" pos="0 0 -1" size="1 1 0.1" type="plane"/>
                <body name="box" pos="0 0 0">
                    <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
                    <joint name="box_joint" type="free"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        # Test model creation
        model = mujoco.MjModel.from_xml_string(simple_xml)
        data = mujoco.MjData(model)
        print("✅ MuJoCo model creation successful")
        
        # Test simulation step
        mujoco.mj_step(model, data)
        print("✅ MuJoCo simulation step successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ MuJoCo import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MuJoCo functionality test failed: {e}")
        return False


def check_model_file():
    """Check if the TVC model file exists"""
    print("\n" + "="*50)
    print("Checking TVC Model File")
    print("="*50)
    
    model_file = "athena_tvc.xml"
    if os.path.exists(model_file):
        print(f"✅ Model file found: {model_file}")
        
        # Check file size (should be reasonable)
        file_size = os.path.getsize(model_file)
        if file_size > 1000:  # At least 1KB
            print(f"✅ Model file size: {file_size} bytes")
            return True
        else:
            print(f"❌ Model file seems too small: {file_size} bytes")
            return False
    else:
        print(f"❌ Model file not found: {model_file}")
        print("   Make sure athena_tvc.xml is in the current directory")
        return False


def test_tvc_simulation():
    """Test the TVC simulation"""
    print("\n" + "="*50)
    print("Testing TVC Simulation")
    print("="*50)
    
    try:
        # Import TVC simulation
        from mujoco_tvc_sim import MuJoCoTVCSimulation
        print("✅ TVC simulation import successful")
        
        # Create simulation instance
        sim = MuJoCoTVCSimulation()
        print("✅ TVC simulation creation successful")
        
        # Test basic functionality
        state = sim.get_rocket_state()
        print(f"✅ State reading successful: position {state['position']}")
        
        # Test a single simulation step
        sim.step_simulation()
        print("✅ Simulation step successful")
        
        print(f"✅ TVC simulation is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ TVC simulation import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ TVC simulation test failed: {e}")
        return False


def run_setup():
    """Run the complete setup process"""
    print("Athena TVC MuJoCo Simulation Setup")
    print("="*50)
    
    setup_success = True
    
    # Check Python version
    if not check_python_version():
        setup_success = False
        return setup_success
    
    # Install requirements
    if not install_requirements():
        setup_success = False
        print("❌ Some packages failed to install")
    
    # Verify MuJoCo
    if not verify_mujoco_installation():
        setup_success = False
        print("❌ MuJoCo verification failed")
    
    # Check model file
    if not check_model_file():
        setup_success = False
        print("❌ Model file check failed")
    
    # Test TVC simulation
    if setup_success and not test_tvc_simulation():
        setup_success = False
        print("❌ TVC simulation test failed")
    
    # Final status
    print("\n" + "="*50)
    print("Setup Complete")
    print("="*50)
    
    if setup_success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python test_mujoco_tvc.py' to test the simulation")
        print("2. Run 'python examples_mujoco_tvc.py' for demonstrations")
        print("3. Run 'python mujoco_tvc_sim.py' for a quick demo")
        print("\nFor interactive visualization:")
        print("- Use sim.run_interactive_simulation() in your code")
        print("- Or run the interactive example from examples_mujoco_tvc.py")
    else:
        print("❌ Setup encountered issues. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Ensure you have Python 3.8+ installed")
        print("2. Try upgrading pip: python -m pip install --upgrade pip")
        print("3. For MuJoCo issues, check: https://mujoco.readthedocs.io/")
        print("4. Make sure athena_tvc.xml is in the current directory")
    
    return setup_success


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Athena TVC MuJoCo Simulation")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check existing installation without installing")
    parser.add_argument("--force-install", action="store_true",
                       help="Force reinstall all packages")
    
    args = parser.parse_args()
    
    if args.check_only:
        print("Checking existing installation...")
        check_python_version()
        
        packages = ["numpy", "matplotlib", "mujoco", "imageio", "cv2"]
        for pkg in packages:
            check_package(pkg, pkg)
        
        verify_mujoco_installation()
        check_model_file()
        test_tvc_simulation()
    
    elif args.force_install:
        print("Force reinstalling all packages...")
        packages = [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0", 
            "mujoco>=2.3.0",
            "imageio>=2.9.0",
            "opencv-python>=4.5.0"
        ]
        
        for pkg in packages:
            install_package(pkg)
        
        verify_mujoco_installation()
        check_model_file()
        test_tvc_simulation()
    
    else:
        run_setup()


if __name__ == "__main__":
    main() 