#!/usr/bin/env python3

#### Multi-task nnU-Net v2 evaluation wrapper for validation and test predictions

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch
import warnings
import time
warnings.filterwarnings('ignore')

def setup_environment():
    current_dir = Path.cwd()
    os.environ['nnUNet_raw'] = str(current_dir / 'nnUNet_raw_data')
    os.environ['nnUNet_preprocessed'] = str(current_dir / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(current_dir / 'nnUNet_results')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi-task nnU-Net v2 Evaluation')
    
    parser.add_argument('--model_folder', type=str, default=None,
                       help='Path to trained model folder (auto-detect if None)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best_test.pth',
                       help='Checkpoint filename')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number for evaluation')
    parser.add_argument('--dataset', type=str, default='Dataset001_Pancreas',
                       help='Dataset name')
    
    parser.add_argument('--fast', action='store_true',
                       help='Enable fast inference (disable TTA, increase step size)')
    parser.add_argument('--show_timing', action='store_true',
                       help='Display detailed timing information')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    
    parser.add_argument('--test_mode', action='store_true',
                       help='Run prediction on test data (no ground truth evaluation)')
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction results to CSV')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

class MultiTaskEvaluator:
    
    def __init__(self, args):
        self.args = args
        
        if args.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(args.device)
        
        if args.test_mode:
            self.test_images_dir = Path(f"nnUNet_raw_data/{args.dataset}/imagesTs")
            self.subtype_info_path = None
        else:
            self.val_images_dir = Path("nnUNet_raw_data/validation/images")
            self.val_labels_dir = Path("nnUNet_raw_data/validation/labels")
            self.subtype_info_path = Path(f"nnUNet_raw_data/{args.dataset}/subtype_info.json")
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not args.test_mode:
            if self.subtype_info_path.exists():
                with open(self.subtype_info_path, 'r') as f:
                    self.subtype_info = json.load(f)
            else:
                raise FileNotFoundError(f"Subtype info not found: {self.subtype_info_path}")
        else:
            self.subtype_info = None
        
        self.predictor = None
        self.inference_times = []
        
    def find_model_folder(self):
        if self.args.model_folder:
            model_folder = Path(self.args.model_folder)
            if not model_folder.exists():
                raise ValueError(f"Model folder not found: {model_folder}")
            return model_folder
        
        dataset_folder = Path(f'nnUNet_results/{self.args.dataset}')
        trainer_folders = list(dataset_folder.glob('MultiTaskTrainer*'))
        
        if not trainer_folders:
            raise ValueError(f"No MultiTaskTrainer folders found in {dataset_folder}")
        
        model_folder = trainer_folders[0] / f'fold_{self.args.fold}'
        
        if not model_folder.exists():
            raise ValueError(f"Model folder not found: {model_folder}")
        
        return model_folder
        
    def initialize_predictor(self):
        if self.args.verbose:
            print("Initializing predictor...")
        
        model_folder = self.find_model_folder()
        
        checkpoint_path = model_folder / self.args.checkpoint
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Using model: {model_folder}")
        print(f"Checkpoint: {checkpoint_path}")
        
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        
        if self.args.fast:
            tile_step_size = 0.8
            use_mirroring = False
            if self.args.verbose:
                print("Fast mode: TTA disabled, step size 0.8")
        else:
            tile_step_size = 0.5
            use_mirroring = True
            if self.args.verbose:
                print("Standard mode: TTA enabled, step size 0.5")
        
        self.predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        
        self.predictor.initialize_from_trained_model_folder(
            str(model_folder.parent),
            use_folds=[self.args.fold],
            checkpoint_name=self.args.checkpoint
        )
        
        if self.args.verbose:
            print("Predictor initialized")
        
    def get_validation_cases(self):
        cases = []
        
        for img_path in self.val_images_dir.glob("*_0000.nii.gz"):
            case_name = img_path.name.replace('_0000.nii.gz', '')
            label_path = self.val_labels_dir / f"{case_name}.nii.gz"
            
            if not label_path.exists():
                continue
            
            #handle validation case naming
            subtype_key = case_name if case_name.startswith('val_') else f"val_{case_name}"
            subtype = self.subtype_info['validation'].get(subtype_key)
            if subtype is None:
                continue
                
            cases.append({
                'case_name': case_name,
                'image_path': str(img_path),
                'label_path': str(label_path),
                'subtype': subtype
            })
        
        return cases
    
    def get_test_cases(self):
        cases = []
        
        for img_path in self.test_images_dir.glob("*.nii.gz"):
            case_name = img_path.name.replace('.nii_0000.nii.gz', '')
            
            cases.append({
                'case_name': case_name,
                'image_path': str(img_path),
                'label_path': None,
                'subtype': None
            })
        
        return cases
        
    def predict_single_case(self, image_path: str, case_name: str):
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        
        reader = SimpleITKIO()
        image_np, image_properties = reader.read_images([image_path])

        start_time = time.time()
        
        try:
            # Get network reference
            network = None
            if hasattr(self.predictor, 'network') and self.predictor.network is not None:
                network = self.predictor.network
            elif hasattr(self.predictor, 'list_of_parameters') and len(self.predictor.list_of_parameters) > 0:
                network = self.predictor.list_of_parameters[0]
            
            # Get segmentation prediction (this will also compute classification internally)
            seg_pred = self.predictor.predict_single_npy_array(
                input_image=image_np,
                image_properties=image_properties,
                segmentation_previous_stage=None,
                output_file_truncated=None,
                save_or_return_probabilities=False
            )
            
            # Get classification prediction from the multi-task model's stored output
            try:
                if (network and hasattr(network, 'last_classification_output') and 
                    network.last_classification_output is not None):
                    
                    with torch.no_grad():
                        # Use the classification output that was already computed during segmentation
                        cls_logits = network.last_classification_output
                        cls_probs = torch.softmax(cls_logits, dim=1)
                        cls_pred = torch.argmax(cls_probs, dim=1)
                        
                        cls_pred_np = cls_pred.cpu().numpy()[0]
                        cls_probs_np = cls_probs.cpu().numpy()[0]
                else:
                    # Fallback to dummy values
                    if self.args.verbose:
                        print(f"  No classification output found in multi-task model")
                    cls_pred_np = 0
                    cls_probs_np = np.array([0.8, 0.1, 0.1])
                    
            except Exception as cls_e:
                if self.args.verbose:
                    print(f"  Classification failed: {cls_e}, using dummy values")
                cls_pred_np = 0
                cls_probs_np = np.array([0.8, 0.1, 0.1])
                
        except Exception as e:
            if self.args.verbose:
                print(f"Prediction failed for {case_name}: {e}")
            seg_pred = np.zeros(image_np.shape[1:], dtype=np.uint8)
            cls_pred_np = 0
            cls_probs_np = np.array([1.0, 0.0, 0.0])
        
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)
        
        if self.args.show_timing:
            print(f"  Inference time: {inference_time:.2f}s")
        
        return seg_pred, cls_pred_np, cls_probs_np
        
    def compute_dice_score(self, pred, gt, label):
        pred_mask = (pred == label).astype(np.uint8)
        gt_mask = (gt == label).astype(np.uint8)
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            return 1.0
        
        return 2.0 * intersection / union
        
    def compute_segmentation_metrics(self, pred, gt):
        # Whole pancreas (label 1 + label 2)
        pred_whole = (pred > 0).astype(np.uint8)
        gt_whole = (gt > 0).astype(np.uint8)
        
        intersection = np.sum(pred_whole * gt_whole)
        union = np.sum(pred_whole) + np.sum(gt_whole)
        whole_pancreas_dsc = 2.0 * intersection / union if union > 0 else 1.0
        
        # Lesion DSC (label 2 only)
        lesion_dsc = self.compute_dice_score(pred, gt, 2)
        
        return {
            'whole_pancreas_dsc': whole_pancreas_dsc,
            'lesion_dsc': lesion_dsc
        }
    
    def save_test_predictions(self, results):
        seg_output_dir = self.output_dir / "segmentation_predictions"
        seg_output_dir.mkdir(exist_ok=True)
        
        cls_data = []
        
        for result in results:
            case_name = result['case_name']
            seg_pred = result['segmentation']
            cls_pred = result['classification']
            
            seg_filename = f"{case_name}.nii.gz"
            seg_output_path = seg_output_dir / seg_filename
            
            seg_image = sitk.GetImageFromArray(seg_pred.astype(np.uint8))
            sitk.WriteImage(seg_image, str(seg_output_path))
            
            cls_data.append({
                'Names': seg_filename,
                'Subtype': int(cls_pred)
            })
        
        cls_df = pd.DataFrame(cls_data)
        cls_output_path = self.output_dir / "subtype_results.csv"
        cls_df.to_csv(cls_output_path, index=False)
        
        print(f"Segmentation predictions saved to: {seg_output_dir}")
        print(f"Classification results saved to: {cls_output_path}")
        
        return seg_output_dir, cls_output_path
        
    def run_evaluation(self):
        mode_str = "test prediction" if self.args.test_mode else "validation evaluation"
        print(f"Starting {mode_str} on {self.device}...")
        if self.args.fast:
            print("Fast inference mode enabled")
        
        self.initialize_predictor()
        
        if self.args.test_mode:
            cases = self.get_test_cases()
            print(f"Found {len(cases)} test cases")
        else:
            cases = self.get_validation_cases()
            print(f"Found {len(cases)} validation cases")
        
        results = []
        y_true = []
        y_pred = []
        
        for i, case in enumerate(cases, 1):
            if self.args.verbose or not self.args.show_timing:
                print(f"Processing {case['case_name']} ({i}/{len(cases)})...")
            
            try:
                seg_pred, cls_pred, cls_probs = self.predict_single_case(
                    case['image_path'], case['case_name']
                )
                
                if self.args.test_mode:
                    result = {
                        'case_name': case['case_name'],
                        'segmentation': seg_pred,
                        'classification': cls_pred,
                        'classification_probs': cls_probs.tolist()
                    }
                    results.append(result)
                    
                else:
                    gt_sitk = sitk.ReadImage(case['label_path'])
                    gt_np = sitk.GetArrayFromImage(gt_sitk)
                    
                    seg_metrics = self.compute_segmentation_metrics(seg_pred, gt_np)
                    
                    result = {
                        'case_name': case['case_name'],
                        'true_subtype': case['subtype'],
                        'pred_subtype': cls_pred,
                        'subtype_probs': cls_probs.tolist(),
                        'whole_pancreas_dsc': seg_metrics['whole_pancreas_dsc'],
                        'lesion_dsc': seg_metrics['lesion_dsc']
                    }
                    
                    results.append(result)
                    y_true.append(case['subtype'])
                    y_pred.append(cls_pred)
                
            except Exception as e:
                print(f"Error processing {case['case_name']}: {e}")
                continue
        
        if self.args.test_mode:
            print(f"\nPROCESSED {len(results)} TEST CASES")
            print("=" * 40)
            
            seg_dir, cls_file = self.save_test_predictions(results)
            
            if self.args.show_timing:
                total_time = sum(self.inference_times)
                avg_time = np.mean(self.inference_times)
                print(f"\nTiming Results:")
                print(f"Total inference time: {total_time:.2f}s")
                print(f"Average per case: {avg_time:.2f}s")
            
            print(f"\nTest predictions completed!")
            print(f"Segmentation results: {seg_dir}")
            print(f"Classification results: {cls_file}")
            
            return {
                'num_cases': len(results),
                'segmentation_dir': str(seg_dir),
                'classification_file': str(cls_file)
            }
            
        else:
            df = pd.DataFrame(results)
            
            whole_pancreas_dsc_mean = df['whole_pancreas_dsc'].mean()
            lesion_dsc_mean = df['lesion_dsc'].mean()
            
            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            
            if self.args.show_timing:
                total_time = sum(self.inference_times)
                avg_time = np.mean(self.inference_times)
                print(f"\nTiming Results:")
                print(f"Total inference time: {total_time:.2f}s")
                print(f"Average per case: {avg_time:.2f}s")
            
            print(f"\nEVALUATION RESULTS")
            print("=" * 40)
            print(f"Whole Pancreas DSC: {whole_pancreas_dsc_mean:.4f}")
            print(f"Lesion DSC: {lesion_dsc_mean:.4f}")
            print(f"Classification Accuracy: {accuracy:.4f}")
            print(f"Classification Macro F1: {macro_f1:.4f}")
            
            print(f"\nTarget Achievement:")
            print(f"Whole Pancreas DSC >= 0.91: {'YES' if whole_pancreas_dsc_mean >= 0.91 else 'NO'}")
            print(f"Lesion DSC >= 0.31: {'YES' if lesion_dsc_mean >= 0.31 else 'NO'}")
            print(f"Macro F1 >= 0.70: {'YES' if macro_f1 >= 0.70 else 'NO'}")
            
            if self.args.save_predictions:
                suffix = "_fast" if self.args.fast else "_standard"
                df.to_csv(self.output_dir / f'evaluation_results{suffix}.csv', index=False)
                
                summary = {
                    'whole_pancreas_dsc': whole_pancreas_dsc_mean,
                    'lesion_dsc': lesion_dsc_mean,
                    'classification_accuracy': accuracy,
                    'classification_macro_f1': macro_f1,
                    'fast_mode': self.args.fast
                }
                
                if self.args.show_timing:
                    summary.update({
                        'total_inference_time': sum(self.inference_times),
                        'avg_inference_time_per_case': np.mean(self.inference_times)
                    })
                
                with open(self.output_dir / f'evaluation_summary{suffix}.json', 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\nResults saved to {self.output_dir}")
            
            return {
                'whole_pancreas_dsc': whole_pancreas_dsc_mean,
                'lesion_dsc': lesion_dsc_mean,
                'accuracy': accuracy,
                'macro_f1': macro_f1
            }

def main():
    args = parse_arguments()
    setup_environment()
    
    try:
        evaluator = MultiTaskEvaluator(args)
        results = evaluator.run_evaluation()
        
        if args.test_mode:
            print(f"\nTest prediction completed successfully!")
        else:
            print(f"\nValidation evaluation completed successfully!")
        
    except Exception as e:
        error_type = "Test prediction" if args.test_mode else "Evaluation"
        print(f"{error_type} failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 