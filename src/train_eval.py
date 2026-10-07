import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from src.data_loader import load_data, get_train_test_split, get_kfold_splits
from src.models import get_model_by_name

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """Calculate key medical classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(spec),
        'f1': float(f1),
        'roc_auc': float(auc),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

def run_kfold_experiment(X_train_val, y_train_val, model_names, epochs=70, batch_size=32, n_splits=5):
    """
    Run 5-Fold Stratified Cross Validation across all model architectures.
    """
    skf = get_kfold_splits(n_splits=n_splits)
    results = {}
    
    for name in model_names:
        print(f"\n--- Running 5-Fold CV for {name} ---")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val), 1):
            X_tr, y_tr = X_train_val[train_idx], y_train_val[train_idx]
            X_va, y_va = X_train_val[val_idx], y_train_val[val_idx]
            
            model = get_model_by_name(name, input_dim=X_train_val.shape[1], learning_rate=0.001)
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=12, restore_best_weights=True, verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0
                )
            ]
            
            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            y_val_prob = model.predict(X_va, verbose=0).ravel()
            metrics = evaluate_predictions(y_va, y_val_prob)
            fold_metrics.append(metrics)
            print(f"  Fold {fold}: Acc={metrics['accuracy']:.4f}, Rec={metrics['recall']:.4f}, AUC={metrics['roc_auc']:.4f}")
            
        # Compute mean & std
        avg_metrics = {}
        for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']:
            vals = [m[k] for m in fold_metrics]
            avg_metrics[k + '_mean'] = float(np.mean(vals))
            avg_metrics[k + '_std'] = float(np.std(vals))
            
        results[name] = {
            'folds': fold_metrics,
            'summary': avg_metrics
        }
    
    return results

def run_optimizer_experiment(X_train_val, y_train_val, optimizers=['adam', 'rmsprop', 'sgd'], epochs=70):
    """
    Analyze convergence & learning curve behaviors across optimizers on Regularized MLP.
    """
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    opt_histories = {}
    for opt in optimizers:
        print(f"\nTraining with optimizer: {opt}")
        lr = 0.001 if opt != 'sgd' else 0.01
        model = get_model_by_name('Regularized_MLP', input_dim=X_train_val.shape[1], learning_rate=lr, optimizer_name=opt)
        
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        opt_histories[opt] = {
            'loss': hist.history['loss'],
            'val_loss': hist.history['val_loss'],
            'accuracy': hist.history['accuracy'],
            'val_accuracy': hist.history['val_accuracy']
        }
        
    return opt_histories

def run_learning_rate_experiment(X_train_val, y_train_val, learning_rates=[0.01, 0.001, 0.0001], epochs=60):
    """
    Analyze effect of learning rate on convergence.
    """
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    lr_histories = {}
    for lr in learning_rates:
        print(f"Training with lr: {lr}")
        model = get_model_by_name('Residual_MLP', input_dim=X_train_val.shape[1], learning_rate=lr)
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        lr_histories[str(lr)] = {
            'loss': hist.history['loss'],
            'val_loss': hist.history['val_loss'],
            'accuracy': hist.history['accuracy'],
            'val_accuracy': hist.history['val_accuracy']
        }
    return lr_histories

def train_and_evaluate_test(X_train_val, y_train_val, X_test, y_test, model_names, epochs=70, batch_size=32):
    """
    Train final models on all train_val data and evaluate on the unseen holdout test set.
    """
    test_results = {}
    histories = {}
    
    for name in model_names:
        print(f"\nFinal training & evaluation on Test Set for: {name}")
        model = get_model_by_name(name, input_dim=X_train_val.shape[1], learning_rate=0.001)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0
            )
        ]
        
        # Split a 15% validation slice from train_val for early stopping
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
        )
        
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predict on holdout test set
        y_test_prob = model.predict(X_test, verbose=0).ravel()
        metrics = evaluate_predictions(y_test, y_test_prob)
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        
        test_results[name] = {
            'metrics': metrics,
            'y_pred_prob': y_test_prob.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        histories[name] = {
            'loss': hist.history['loss'],
            'val_loss': hist.history['val_loss'],
            'accuracy': hist.history['accuracy'],
            'val_accuracy': hist.history['val_accuracy']
        }
        print(f"  {name} Test Result -> Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
    return test_results, histories

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    df, X, y, feat_names = load_data()
    X_tr_val, X_te, y_tr_val, y_te, scaler = get_train_test_split(X, y)
    
    models = ['Baseline_MLP', 'Deep_MLP', 'Regularized_MLP', 'Residual_MLP']
    
    # 1. K-Fold CV
    kfold_res = run_kfold_experiment(X_tr_val, y_tr_val, models)
    with open('results/kfold_results.json', 'w') as f:
        json.dump(kfold_res, f, indent=2)
        
    # 2. Optimizer tuning
    opt_res = run_optimizer_experiment(X_tr_val, y_tr_val)
    with open('results/optimizer_results.json', 'w') as f:
        json.dump(opt_res, f, indent=2)
        
    # 3. Learning Rate tuning
    lr_res = run_learning_rate_experiment(X_tr_val, y_tr_val)
    with open('results/lr_results.json', 'w') as f:
        json.dump(lr_res, f, indent=2)
        
    # 4. Final test set evaluation
    test_res, train_hists = train_and_evaluate_test(X_tr_val, y_tr_val, X_te, y_te, models)
    with open('results/test_results.json', 'w') as f:
        json.dump(test_res, f, indent=2)
    with open('results/training_histories.json', 'w') as f:
        json.dump(train_hists, f, indent=2)
        
    print("\nAll experiments successfully completed and saved in results/!")
