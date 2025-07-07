#!/usr/bin/env python3

# Multi-task nnU-Net v2 training script with ResidualEncoderUNet architecture

import os
import sys
import argparse
from pathlib import Path
import json
import torch

def setup_environment():
    current_dir = Path.cwd()
    env_vars = {
        'nnUNet_raw': str(current_dir / 'nnUNet_raw_data'),
        'nnUNet_preprocessed': str(current_dir / 'nnUNet_preprocessed'),
        'nnUNet_results': str(current_dir / 'nnUNet_results')
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Limit threading for HPC compatibility
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 
                'OPENBLAS_NUM_THREADS', 'BLAS_NUM_THREADS', 'LAPACK_NUM_THREADS']:
        os.environ[var] = '1'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi-task nnU-Net v2 Training')
    
    parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold')
    parser.add_argument('--configuration', type=str, default='3d_fullres', 
                       help='nnU-Net configuration')
    parser.add_argument('--dataset', type=str, default='Dataset001_Pancreas',
                       help='Dataset name')
    
    parser.add_argument('--cls_loss_weight', type=float, default=100.0,
                       help='Classification loss weight')
    parser.add_argument('--num_classes_cls', type=int, default=3,
                       help='Number of classification classes')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='Use focal loss for classification')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose logging')
    
    return parser.parse_args()

def verify_preprocessing(dataset_name):
    required_files = [
        f'nnUNet_preprocessed/{dataset_name}/nnUNetPlans.json',
        f'nnUNet_preprocessed/{dataset_name}/dataset.json',
        f'nnUNet_raw_data/{dataset_name}/dataset.json',
        f'nnUNet_raw_data/{dataset_name}/subtype_info.json'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def load_plans_and_dataset(dataset_name):
    preprocessed_dir = Path(f'nnUNet_preprocessed/{dataset_name}')
    
    plans_file = preprocessed_dir / 'nnUNetPlans.json'
    if not plans_file.exists():
        raise FileNotFoundError(f"nnUNetPlans.json not found at {plans_file}")
    
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans_manager = PlansManager(plans_file)
    plans = plans_manager.plans
    
    dataset_json_file = preprocessed_dir / 'dataset.json'
    if not dataset_json_file.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json_file}")
    
    with open(dataset_json_file, 'r') as f:
        dataset_json = json.load(f)
    
    return plans, dataset_json

def analyze_class_distribution(dataset_name, verbose=False):
    if not verbose:
        return
    
    subtype_file = Path(f'nnUNet_raw_data/{dataset_name}/subtype_info.json')
    if not subtype_file.exists():
        print("Warning: subtype_info.json not found")
        return
    
    with open(subtype_file, 'r') as f:
        subtype_info = json.load(f)
    
    print("\nClass Distribution:")
    
    if 'training' in subtype_info:
        train_labels = list(subtype_info['training'].values())
        train_counts = {0: 0, 1: 0, 2: 0}
        for label in train_labels:
            train_counts[label] += 1
        
        total_train = sum(train_counts.values())
        print(f"Training: {total_train} cases")
        for class_id in [0, 1, 2]:
            count = train_counts[class_id]
            percentage = (count / total_train) * 100
            print(f"  Class {class_id}: {count:3d} ({percentage:5.1f}%)")

def train_multitask_model(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Training on device: {device}")
    
    plans, dataset_json = load_plans_and_dataset(args.dataset)
    
    from nnunetv2.training.nnUNetTrainer.multitask_trainer import MultiTaskTrainer
    
    trainer = MultiTaskTrainer(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        unpack_dataset=True,
        device=device
    )
    
    trainer.cls_loss_weight = args.cls_loss_weight
    trainer.num_classes_cls = args.num_classes_cls
    trainer.use_focal_loss = args.use_focal_loss
    
    if args.num_epochs is not None:
        trainer.num_epochs = args.num_epochs
    
    print(f"Output folder: {trainer.output_folder}")
    print(f"Configuration: {args.configuration}")
    print(f"Fold: {args.fold}")
    print(f"Classification loss weight: {args.cls_loss_weight}")
    print(f"Number of epochs: {trainer.num_epochs}")
    
    trainer.run_training()
    
    # Save model for evaluation
    final_model_path = Path(trainer.output_folder) / "final_model.pth"
    torch.save({
        'network_weights': trainer.network.state_dict(),
        'trainer_state': {
            'cls_loss_weight': trainer.cls_loss_weight,
            'num_classes_cls': trainer.num_classes_cls,
            'use_focal_loss': trainer.use_focal_loss
        }
    }, final_model_path)
    
    print(f"\nTraining completed! Results saved to: {trainer.output_folder}")
    print(f"Final model saved to: {final_model_path}")
    return trainer.output_folder

def main():
    args = parse_arguments()
    setup_environment()
    
    print("Multi-task nnU-Net v2 Training")
    print("=" * 40)
    
    if not verify_preprocessing(args.dataset):
        print("Error: Preprocessing not completed")
        print("Please run preprocessing first")
        sys.exit(1)
    
    analyze_class_distribution(args.dataset, args.verbose)
    
    try:
        output_folder = train_multitask_model(args)
        print(f"\nTraining completed successfully!")
        print(f"Results: {output_folder}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 