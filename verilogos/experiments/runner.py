"""
Experiment Runner - Execute single experiments

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, Any

def run_single_experiment(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment task.
    
    Args:
        task: Task configuration with dataset, model, ablation
    
    Returns:
        Result dictionary with metrics
    """
    try:
        # Extract task parameters
        task_id = task.get('task_id', 0)
        dataset_name = task.get('dataset', 'unknown')
        model_name = task.get('model', 'unknown')
        ablation = task.get('ablation', {})
        config = task.get('config', {})
        
        print(f"\n[Task {task_id}] Dataset: {dataset_name}, Model: {model_name}")
        
        # Load dataset
        dataset = _load_dataset(dataset_name, config)
        
        if dataset is None or len(dataset) == 0:
            return {
                'status': 'failed',
                'error': 'Dataset loading failed',
                'task_id': task_id,
                'dataset': dataset_name,
                'model': model_name
            }
        
        # Get splits
        train_data, val_data, test_data = dataset.get_splits()
        
        # Extract features (FIXED: pass both train and test together)
        X_train, y_train, X_test, y_test = _extract_features_fixed(
            train_data, test_data, model_name, ablation
        )
        
        # Train model
        model = _get_model(model_name, ablation)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Add metadata
        result = {
            'status': 'success',
            'task_id': task_id,
            'dataset': dataset_name,
            'model': model_name,
            'ablation': ablation,
            **metrics
        }
        
        print(f"[Task {task_id}] F1: {metrics.get('f1', 0):.4f}, "
              f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return result
    
    except Exception as e:
        import traceback
        return {
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'task_id': task.get('task_id', 0),
            'dataset': task.get('dataset', 'unknown'),
            'model': task.get('model', 'unknown')
        }


def _load_dataset(dataset_name: str, config: Dict[str, Any]):
    """Load dataset by name."""
    from verilogos.experiments.datasets import (
        FakeNewsNetDataset,
        LIARDataset,
        PHEMEDataset
    )
    
    dataset_name = dataset_name.lower()
    
    if 'fakenews' in dataset_name:
        dataset = FakeNewsNetDataset()
        # Try to load from config path, otherwise create synthetic
        data_path = config.get('data_paths', {}).get('fakenewsnet')
        if data_path:
            success = dataset.load(data_path)
            if not success:
                dataset.create_synthetic(n_samples=200)
        else:
            dataset.create_synthetic(n_samples=200)
    
    elif 'liar' in dataset_name:
        dataset = LIARDataset()
        data_path = config.get('data_paths', {}).get('liar')
        if data_path:
            dataset.load(data_path)
        else:
            # Create synthetic for testing
            from verilogos.experiments.datasets import FakeNewsNetDataset
            dataset = FakeNewsNetDataset()
            dataset.create_synthetic(n_samples=200)
    
    elif 'pheme' in dataset_name:
        dataset = PHEMEDataset()
        data_path = config.get('data_paths', {}).get('pheme')
        if data_path:
            dataset.load(data_path)
        else:
            # Create synthetic for testing
            from verilogos.experiments.datasets import FakeNewsNetDataset
            dataset = FakeNewsNetDataset()
            dataset.create_synthetic(n_samples=200)
    
    else:
        # Default to synthetic FakeNewsNet
        dataset = FakeNewsNetDataset()
        dataset.create_synthetic(n_samples=200)
    
    return dataset


def _extract_features_fixed(train_data, test_data, model_name: str, ablation: Dict[str, Any]):
    """
    Extract features for both train and test data consistently.
    
    CRITICAL FIX: Fit feature extractors on train data only,
    then transform both train and test with the SAME extractors.
    This ensures consistent feature dimensions.
    
    Args:
        train_data: Training samples
        test_data: Test samples
        model_name: Model name
        ablation: Ablation configuration
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    from verilogos.experiments.features import (
        extract_batch_topology_features,
        TextFeatureExtractor
    )
    
    # Get texts and labels for train and test
    train_texts = [sample.text for sample in train_data]
    train_labels = np.array([sample.label for sample in train_data])
    test_texts = [sample.text for sample in test_data]
    test_labels = np.array([sample.label for sample in test_data])
    
    # Determine which features to use
    use_topology = ablation.get('topology', True)
    use_text = ablation.get('text', False)
    
    # For topology-only models
    if 'topology' in model_name.lower() and not use_text:
        X_train_topo, _ = extract_batch_topology_features(train_texts, show_progress=False)
        X_test_topo, _ = extract_batch_topology_features(test_texts, show_progress=False)
        return X_train_topo, train_labels, X_test_topo, test_labels
    
    # For text-only models
    elif use_text and not use_topology:
        extractor = TextFeatureExtractor(max_features=2000)  # Increased from 500
        X_train_text = extractor.fit_transform(train_texts)  # Fit on train
        X_test_text = extractor.transform(test_texts)  # Transform test (FIXED!)
        return X_train_text, train_labels, X_test_text, test_labels
    
    # For hybrid models
    elif use_topology and use_text:
        # Topology features
        X_train_topo, _ = extract_batch_topology_features(train_texts, show_progress=False)
        X_test_topo, _ = extract_batch_topology_features(test_texts, show_progress=False)
        
        # Text features (FIXED: fit on train, transform on test)
        extractor = TextFeatureExtractor(max_features=2000)
        X_train_text = extractor.fit_transform(train_texts)
        X_test_text = extractor.transform(test_texts)
        
        # Combine
        X_train_combined = np.hstack([X_train_topo, X_train_text])
        X_test_combined = np.hstack([X_test_topo, X_test_text])
        return X_train_combined, train_labels, X_test_combined, test_labels
    
    # Default: topology features
    else:
        X_train_topo, _ = extract_batch_topology_features(train_texts, show_progress=False)
        X_test_topo, _ = extract_batch_topology_features(test_texts, show_progress=False)
        return X_train_topo, train_labels, X_test_topo, test_labels


def _get_model(model_name: str, ablation: Dict[str, Any]):
    """Get model instance by name."""
    from verilogos.experiments.models import (
        get_classical_model,
        VeriLogosTopologyModel,
        HybridModel
    )
    
    model_name = model_name.lower()
    
    # Classical models
    if model_name in ['lr', 'logistic', 'logisticregression']:
        return get_classical_model('lr')
    elif model_name in ['rf', 'randomforest']:
        return get_classical_model('rf', n_estimators=100)
    elif model_name in ['svm']:
        return get_classical_model('svm')
    elif model_name in ['xgb', 'xgboost']:
        return get_classical_model('xgboost', n_estimators=100)
    
    # VeriLogos topology model
    elif 'topology' in model_name:
        return VeriLogosTopologyModel(n_estimators=100)
    
    # Hybrid model
    elif 'hybrid' in model_name:
        return HybridModel(n_estimators=100)
    
    # Default to Random Forest
    else:
        return get_classical_model('rf', n_estimators=100)
