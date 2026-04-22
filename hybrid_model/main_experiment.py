"""
Main Experiment Orchestrator
Trains and evaluates TDA-only, Text-only, and Hybrid models
Generates publication-ready comparative analysis
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
import sys
sys.path.append('/mnt/d/Verilogos')

from data_loaders import (
    load_acl2017_dataset,
    extract_tda_features_parallel,
    create_dataloaders
)
from hybrid_model import create_model
from trainer import Trainer, evaluate_model
from visualizer import generate_all_visualizations


def run_experiment(
    mode: str,
    train_loader,
    val_loader,
    test_loader,
    device: str,
    num_epochs: int = 20,
    learning_rate: float = 2e-5,
    checkpoint_dir: str = './checkpoints'
):
    """
    Run complete training and evaluation for one model.
    
    Args:
        mode: 'tda_only', 'text_only', or 'hybrid'
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        device: Device to use
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        (test_results, training_history)
    """
    print("\n" + "="*80)
    print(f"TRAINING {mode.upper()} MODEL")
    print("="*80)
    
    # Create model
    model = create_model(
        mode=mode,
        text_model_name='distilroberta-base',
        device=device,
        dropout=0.3,
        freeze_text_encoder=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=0.01,
        num_epochs=num_epochs,
        patience=5,
        checkpoint_dir=f"{checkpoint_dir}/{mode}"
    )
    
    # Train
    history = trainer.train()
    
    # Evaluate on test set
    print(f"\nEvaluating {mode.upper()} model on test set...")
    test_results = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
    
    return test_results, history


def main():
    parser = argparse.ArgumentParser(description='Hybrid TDA-Text Fake News Detection')
    parser.add_argument('--data_path', type=str, 
                       default='/mnt/d/Verilogos/historical_data/rumor_detection_acl2017',
                       help='Path to ACL2017 dataset')
    parser.add_argument('--max_events', type=int, default=400,
                       help='Maximum number of events to use (None = all)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--n_workers_tda', type=int, default=4,
                       help='Number of workers for TDA extraction')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for model checkpoints')
    parser.add_argument('--modes', nargs='+', default=['tda_only', 'text_only', 'hybrid'],
                       choices=['tda_only', 'text_only', 'hybrid'],
                       help='Which models to train')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Dataset
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    events, source_tweets = load_acl2017_dataset(
        base_path=args.data_path,
        max_events=args.max_events,
        balance_classes=True
    )
    
    # ========================================================================
    # STEP 2: Extract TDA Features (if needed)
    # ========================================================================
    precomputed_tda = None
    
    if any(mode in args.modes for mode in ['tda_only', 'hybrid']):
        print("\n" + "="*80)
        print("STEP 2: EXTRACTING TDA FEATURES")
        print("="*80)
        
        precomputed_tda = extract_tda_features_parallel(
            events=events,
            n_workers=args.n_workers_tda,
            lambda_decay=0.001,
            temporal_window=60.0
        )
        
        # Save TDA features
        tda_save_path = Path(args.output_dir) / 'tda_features.npy'
        np.save(tda_save_path, precomputed_tda)
        print(f"Saved TDA features to {tda_save_path}")
    
    # ========================================================================
    # STEP 3: Train and Evaluate Models
    # ========================================================================
    all_results = {}
    all_histories = {}
    
    for mode in args.modes:
        print("\n" + "="*80)
        print(f"STEP 3.{args.modes.index(mode)+1}: {mode.upper()} MODEL")
        print("="*80)
        
        # Create dataloaders for this mode
        train_loader, val_loader, test_loader = create_dataloaders(
            events=events,
            source_tweets=source_tweets,
            mode=mode,
            batch_size=args.batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            precomputed_tda=precomputed_tda,
            tokenizer_name='distilroberta-base',
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Run experiment
        test_results, history = run_experiment(
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Store results
        model_name = mode.replace('_', ' ').title()
        all_results[model_name] = test_results
        all_histories[model_name] = history
    
    # ========================================================================
    # STEP 4: Generate Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80)
    
    generate_all_visualizations(
        results=all_results,
        training_histories=all_histories,
        output_dir=args.output_dir
    )
    
    # ========================================================================
    # STEP 5: Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    from sklearn.metrics import f1_score
    
    for model_name, result in all_results.items():
        f1 = f1_score(result['labels'], result['predictions'], average='weighted')
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Statistical significance test (if hybrid is included)
    if 'Hybrid' in all_results and 'Text Only' in all_results:
        hybrid_f1 = f1_score(
            all_results['Hybrid']['labels'],
            all_results['Hybrid']['predictions'],
            average='weighted'
        )
        text_f1 = f1_score(
            all_results['Text Only']['labels'],
            all_results['Text Only']['predictions'],
            average='weighted'
        )
        improvement = ((hybrid_f1 - text_f1) / text_f1) * 100
        
        print(f"\n{'='*80}")
        print("HYBRID MODEL IMPROVEMENT")
        print(f"{'='*80}")
        print(f"Text-only F1: {text_f1:.4f}")
        print(f"Hybrid F1:    {hybrid_f1:.4f}")
        print(f"Improvement:  {improvement:+.2f}%")
        
        if improvement > 0:
            print("\n✓ Hybrid model outperforms text-only baseline!")
        else:
            print("\n✗ Hybrid model did not improve over text-only baseline.")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == '__main__':
    main()
