"""
Topological Lie Detector - Complete Experiment Runner

This script executes the full end-to-end pipeline for misinformation detection
using topological data analysis and persistent homology.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
License: MIT
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import pipeline components
try:
    from verilogos.applications.lie_detector.pipeline import LieDetectorPipeline
    from verilogos.applications.lie_detector.data.dataset import FakeNewsDataset, NewsArticle
    from verilogos.applications.lie_detector.data.loader import DataLoader
    from verilogos.applications.lie_detector.evaluation.metrics import EvaluationMetrics
    from verilogos.applications.lie_detector.evaluation.visualization import Visualizer
    from verilogos.applications.lie_detector.evaluation.explainability import ExplainabilityEngine
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Ensure VeriLogos is properly installed and PYTHONPATH is set")
    sys.exit(1)


def create_sample_dataset(n_samples: int = 100) -> FakeNewsDataset:
    """
    Create a synthetic dataset for testing when real data is unavailable.
    
    Args:
        n_samples: Number of samples to generate (half real, half fake)
        
    Returns:
        FakeNewsDataset with synthetic articles
    """
    logger.info(f"Generating {n_samples} synthetic articles...")
    
    # Real news templates (coherent, structured)
    real_templates = [
        "The government announced new economic policies today. Officials stated the measures will take effect next month. Economists predict positive impacts on employment. The central bank supports these initiatives.",
        "Scientists discovered a breakthrough in renewable energy. The research team published findings in a peer-reviewed journal. Experts believe this could reduce carbon emissions. Further testing is planned for next year.",
        "Local authorities reported improved public safety statistics. Crime rates decreased by fifteen percent this quarter. Community programs contributed to the positive trend. Officials plan to expand successful initiatives.",
        "International trade negotiations concluded successfully. Both nations agreed to reduce tariffs on key goods. Business leaders welcomed the agreement. Implementation begins in six months.",
        "Medical researchers identified a promising treatment approach. Clinical trials showed encouraging results. Regulatory approval is expected within two years. Patient advocacy groups expressed optimism."
    ]
    
    # Fake news templates (fragmented, contradictory, sensational)
    fake_templates = [
        "BREAKING: Secret conspiracy revealed! Anonymous sources claim shocking truth. Mainstream media refuses to report. Share before deleted! Experts deny but we know better.",
        "You won't believe what happened next! Incredible discovery changes everything. Scientists baffled. Government hiding the truth. Click to learn more. Time is running out!",
        "Miracle cure discovered! Doctors hate this one simple trick. Big pharma doesn't want you to know. Thousands already benefiting. Act now before banned. Limited time offer!",
        "Shocking revelation about famous person! Insider leaks explosive information. Career over. Scandal rocks industry. More details emerging. Sources say unbelievable things happening.",
        "Economic collapse imminent! Experts predict disaster. Stock market manipulation exposed. Protect your wealth now. Secret elite plan revealed. Don't be a victim. Urgent action required!"
    ]
    
    articles = []
    
    # Generate real articles
    for i in range(n_samples // 2):
        template = real_templates[i % len(real_templates)]
        # Add variation
        sentences = template.split('. ')
        if i % 3 == 0:
            sentences = sentences[:3]  # Shorter version
        elif i % 3 == 1:
            sentences = sentences + [sentences[0]]  # Add repetition
        text = '. '.join(sentences) + '.'
        
        articles.append(NewsArticle(
            id=f"real_{i}",
            title=f"Real News Article {i}",
            text=text,
            label="real",
            source="reliable_source"
        ))
    
    # Generate fake articles
    for i in range(n_samples // 2):
        template = fake_templates[i % len(fake_templates)]
        # Add more fragmentation
        sentences = template.split('. ')
        if i % 2 == 0:
            # Shuffle sentences to create incoherence
            np.random.shuffle(sentences)
        text = '. '.join(sentences) + '.'
        
        articles.append(NewsArticle(
            id=f"fake_{i}",
            title=f"Fake News Article {i}",
            text=text,
            label="fake",
            source="unreliable_source"
        ))
    
    # Shuffle articles
    np.random.shuffle(articles)
    
    dataset = FakeNewsDataset(articles=articles)
    logger.info(f"Generated {len(dataset)} articles ({dataset.get_statistics()})")
    
    return dataset


def load_or_create_dataset(data_path: Optional[str] = None, n_samples: int = 100) -> FakeNewsDataset:
    """
    Load dataset from file or create synthetic data.
    
    Args:
        data_path: Path to FakeNewsNet JSON file (optional)
        n_samples: Number of synthetic samples if no data file provided
        
    Returns:
        FakeNewsDataset ready for training
    """
    if data_path and Path(data_path).exists():
        logger.info(f"Loading dataset from {data_path}...")
        loader = DataLoader()
        try:
            dataset = loader.load_fakenewsnet(data_path)
            logger.info(f"Loaded {len(dataset)} articles from file")
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load data file: {e}")
            logger.info("Falling back to synthetic data generation")
    
    return create_sample_dataset(n_samples)


def train_test_split(dataset: FakeNewsDataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[FakeNewsDataset, FakeNewsDataset]:
    """
    Split dataset into training and test sets.
    
    Args:
        dataset: Full dataset
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    np.random.seed(random_state)
    
    articles = dataset.articles.copy()
    np.random.shuffle(articles)
    
    split_idx = int(len(articles) * (1 - test_size))
    
    train_articles = articles[:split_idx]
    test_articles = articles[split_idx:]
    
    train_dataset = FakeNewsDataset(articles=train_articles)
    test_dataset = FakeNewsDataset(articles=test_articles)
    
    logger.info(f"Split: {len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_dataset, test_dataset


def run_experiment(
    data_path: Optional[str] = None,
    n_samples: int = 100,
    test_size: float = 0.2,
    output_dir: str = "results",
    save_model: bool = True,
    visualize: bool = True
) -> Dict:
    """
    Run complete lie detection experiment.
    
    Args:
        data_path: Path to dataset file (optional)
        n_samples: Number of synthetic samples if no data file
        test_size: Fraction for test set
        output_dir: Directory to save results
        save_model: Whether to save trained model
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary with experiment results
    """
    logger.info("=" * 80)
    logger.info("TOPOLOGICAL LIE DETECTOR - EXPERIMENT")
    logger.info("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset
    logger.info("\n[1/6] Loading dataset...")
    dataset = load_or_create_dataset(data_path, n_samples)
    
    # Step 2: Split data
    logger.info("\n[2/6] Splitting train/test sets...")
    train_dataset, test_dataset = train_test_split(dataset, test_size)
    
    # Step 3: Initialize pipeline
    logger.info("\n[3/6] Initializing pipeline...")
    pipeline = LieDetectorPipeline(
        similarity_threshold=0.3,
        max_dimension=2,
        n_estimators=100
    )
    
    # Step 4: Train pipeline
    logger.info("\n[4/6] Training pipeline...")
    train_texts = [article.text for article in train_dataset.articles]
    train_labels = [1 if article.label == "fake" else 0 for article in train_dataset.articles]
    
    pipeline.train(train_texts, train_labels)
    logger.info("Training complete!")
    
    # Step 5: Evaluate on test set
    logger.info("\n[5/6] Evaluating on test set...")
    test_texts = [article.text for article in test_dataset.articles]
    test_labels = [1 if article.label == "fake" else 0 for article in test_dataset.articles]
    
    predictions = []
    prediction_probs = []
    explanations = []
    
    for text in test_texts:
        result = pipeline.predict(text)
        predictions.append(result['prediction'])
        prediction_probs.append(result['confidence'])
        explanations.append(result['explanation'])
    
    # Calculate metrics
    metrics = EvaluationMetrics()
    results = metrics.compute_all_metrics(test_labels, predictions, prediction_probs)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall:    {results['recall']:.4f}")
    logger.info(f"F1 Score:  {results['f1']:.4f}")
    logger.info(f"ROC-AUC:   {results['roc_auc']:.4f}")
    logger.info("=" * 80)
    
    # Step 6: Generate visualizations and examples
    if visualize:
        logger.info("\n[6/6] Generating visualizations...")
        
        visualizer = Visualizer(output_dir=str(output_path))
        
        # Confusion matrix
        visualizer.plot_confusion_matrix(
            test_labels,
            predictions,
            class_names=['Real', 'Fake'],
            save_path=str(output_path / "confusion_matrix.png")
        )
        logger.info(f"Saved confusion matrix to {output_path / 'confusion_matrix.png'}")
        
        # ROC curve
        visualizer.plot_roc_curve(
            test_labels,
            prediction_probs,
            save_path=str(output_path / "roc_curve.png")
        )
        logger.info(f"Saved ROC curve to {output_path / 'roc_curve.png'}")
        
        # Example predictions with topological visualizations
        logger.info("\nGenerating example predictions with topological analysis...")
        
        for idx in range(min(3, len(test_dataset.articles))):
            article = test_dataset.articles[idx]
            result = pipeline.predict(article.text)
            
            logger.info(f"\n--- Example {idx + 1} ---")
            logger.info(f"True Label: {article.label}")
            logger.info(f"Prediction: {'fake' if result['prediction'] == 1 else 'real'}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info(f"Shape Code: {result['shape_code']}")
            logger.info(f"Explanation: {result['explanation'][:200]}...")
            
            # Generate topological visualizations for this example
            if 'truth_shape' in result and result['truth_shape'] is not None:
                truth_shape = result['truth_shape']
                
                # Persistence diagram
                if hasattr(truth_shape, 'persistence_barcode') and truth_shape.persistence_barcode:
                    visualizer.plot_persistence_diagram(
                        truth_shape.persistence_barcode,
                        save_path=str(output_path / f"persistence_diagram_example_{idx}.png")
                    )
                    
                    # Barcode
                    visualizer.plot_barcode(
                        truth_shape.persistence_barcode,
                        save_path=str(output_path / f"barcode_example_{idx}.png")
                    )
    
    # Save model
    if save_model:
        model_path = output_path / "lie_detector_model.pkl"
        pipeline.save_model(str(model_path))
        logger.info(f"\nSaved trained model to {model_path}")
    
    # Save results to JSON
    results_dict = {
        'metrics': results,
        'dataset_stats': {
            'total_samples': len(dataset),
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'test_size': test_size
        },
        'pipeline_config': {
            'similarity_threshold': 0.3,
            'max_dimension': 2,
            'n_estimators': 100
        }
    }
    
    results_file = output_path / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    
    return results_dict


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Topological Lie Detector - Complete Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to FakeNewsNet JSON dataset file'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of synthetic samples to generate if no data file provided'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results and visualizations'
    )
    
    parser.add_argument(
        '--no-save-model',
        action='store_true',
        help='Do not save trained model'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Do not generate visualizations'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    try:
        results = run_experiment(
            data_path=args.data_path,
            n_samples=args.n_samples,
            test_size=args.test_size,
            output_dir=args.output_dir,
            save_model=not args.no_save_model,
            visualize=not args.no_visualize
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
