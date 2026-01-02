print("Starting test")

if __name__ == '__main__':
    print("In main block")
    try:
        print("About to import numpy")
        import numpy as np
        print("Numpy imported successfully")
        
        print("About to import h5py")
        import h5py
        print("h5py imported successfully")
        
        print("About to import scipy")
        from scipy import integrate
        print("scipy imported successfully")
        
        print("All imports successful!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

