#!/usr/bin/env python3
"""
VeriLogos Experiment Runner

Single command to run all experiments.

Usage:
    python run_experiments.py
    python run_experiments.py --config config/custom.yaml
    python run_experiments.py --sequential

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import argparse
import sys
from pathlib import Path

# Add VeriLogos to path
sys.path.insert(0, str(Path(__file__).parent))

from verilogos.experiments.manager.experiment_manager import ExperimentManager


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VeriLogos NeurIPS-Grade Experiment Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='verilogos/experiments/config/experiments.yaml',
        help='Path to experiment configuration file'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: CPU count - 1)'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run experiments sequentially (for debugging)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Run only specific dataset'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Run only specific model'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print("VeriLogos NeurIPS-Grade Experiment Manager")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Results: {args.results_dir}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print("="*80)
    
    # Initialize manager
    try:
        manager = ExperimentManager(
            config_path=args.config,
            results_dir=args.results_dir,
            max_workers=args.max_workers
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the configuration file exists.")
        print("You can create a default config with:")
        print("  mkdir -p verilogos/experiments/config")
        print("  # Then add experiments.yaml")
        return 1
    
    # Run experiments
    try:
        if args.dataset:
            # Run specific dataset
            results = manager.run_single_dataset(
                args.dataset,
                parallel=not args.sequential
            )
        elif args.model:
            # Run specific model
            results = manager.run_single_model(
                args.model,
                parallel=not args.sequential
            )
        else:
            # Run all experiments
            results = manager.run_all_experiments(
                parallel=not args.sequential
            )
        
        # Export results
        manager.export_results_csv()
        
        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {args.results_dir}")
        print(f"Total experiments: {len(results)}")
        
        successful = sum(1 for r in results if r.get('status') == 'success')
        print(f"Successful: {successful}/{len(results)}")
        
        if successful > 0:
            best_f1 = max(r.get('f1', 0) for r in results if r.get('status') == 'success')
            print(f"Best F1 Score: {best_f1:.4f}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n\nError running experiments: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
