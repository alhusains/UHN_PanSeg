#!/usr/bin/env python3

###Setup nnU-Net v2 environment and run fingerprint, planning, preprocessing
import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    current_dir = Path.cwd()
    
    env_vars = {
        'nnUNet_raw': str(current_dir / 'nnUNet_raw_data'),
        'nnUNet_preprocessed': str(current_dir / 'nnUNet_preprocessed'),
        'nnUNet_results': str(current_dir / 'nnUNet_results')
    }
    os.environ.update(env_vars)
    
    #Limit multiprocessing for HPC compatibility
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 
                'OPENBLAS_NUM_THREADS', 'BLAS_NUM_THREADS', 'LAPACK_NUM_THREADS']:
        os.environ[var] = '1'
    
    #Create directories
    for dir_path in env_vars.values():
        Path(dir_path).mkdir(exist_ok=True)
    
    return env_vars

def verify_dataset():
    task_dir = Path('nnUNet_raw_data/Dataset001_Pancreas')
    if not task_dir.exists():
        print("Error: Dataset001_Pancreas directory not found")
        return False
    
    #Check required subdirectories and files
    required = ['imagesTr', 'labelsTr', 'imagesTs', 'dataset.json']
    for item in required:
        if not (task_dir / item).exists():
            print(f"Error: {item} not found!")
            return False
    
    #verify data exists
    images_count = len(list((task_dir / 'imagesTr').glob('*.nii.gz')))
    labels_count = len(list((task_dir / 'labelsTr').glob('*.nii.gz')))
    
    if images_count == 0 or labels_count == 0:
        print("Error: No training data found")
        return False
    
    return True

def run_command(cmd, timeout=1800):
    env = os.environ.copy()
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 
                'OPENBLAS_NUM_THREADS', 'BLAS_NUM_THREADS', 'LAPACK_NUM_THREADS']:
        env[var] = '1'
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("Setting up nnU-Net v2 environment...")
    
    #Verify dataset
    if not verify_dataset():
        sys.exit(1)
    
    #Setup environment
    setup_environment()
    
    #Run the three steps
    task_id = 1
    
    #Step 1: Extract fingerprint
    print("Extracting dataset fingerprint...")
    if not run_command(['nnUNetv2_extract_fingerprint', '-d', str(task_id), '-np', '1']):
        sys.exit(1)
    
    #Step 2: Plan experiment
    print("Planning experiment...")
    if not run_command(['nnUNetv2_plan_experiment', '-d', str(task_id), 
                       '-pl', 'ExperimentPlanner', '-gpu_memory_target', '8']):
        sys.exit(1)
    
    #Step 3: Preprocess
    print("Running preprocessing...")
    if not run_command(['nnUNetv2_preprocess', '-d', str(task_id), 
                       '-pl', 'nnUNetPlans', '-np', '1'], timeout=3600):
        sys.exit(1)
    
    print("Setup completed successfully")

if __name__ == "__main__":
    main() 