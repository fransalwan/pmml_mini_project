import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Styling configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

os.makedirs('assets', exist_ok=True)

# 1. Figure 1: Data Distribution & Clinical Features
def plot_data_distribution():
    from src.data_loader import load_data
    df, _, _, _ = load_data()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    # Subplot 1: Target distribution
    counts = df['diagnosis'].value_counts()
    colors = ['#2b5c8f', '#d9534f']
    axes[0].bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    for i, v in enumerate(counts.values):
        pct = (v / len(df)) * 100
        axes[0].text(i, v + 8, f"{v} ({pct:.1f}%)", ha='center', fontweight='bold', fontsize=11)
    axes[0].set_title("Distribusi Kelas Diagnosis", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel("Jumlah Pasien", fontsize=11)
    axes[0].set_ylim(0, 420)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Subplot 2: Mean Radius vs Mean Texture
    malignant = df[df['target'] == 1]
    benign = df[df['target'] == 0]
    axes[1].scatter(benign['mean radius'], benign['mean texture'], color='#2b5c8f', alpha=0.7, label='Benign (Jinak)', s=28)
    axes[1].scatter(malignant['mean radius'], malignant['mean texture'], color='#d9534f', alpha=0.8, label='Malignant (Ganas)', s=32, marker='^')
    axes[1].set_title("Mean Radius vs Mean Texture", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Mean Radius", fontsize=11)
    axes[1].set_ylabel("Mean Texture", fontsize=11)
    axes[1].legend(frameon=True, facecolor='white', loc='upper left')
    axes[1].grid(linestyle='--', alpha=0.5)
    
    # Subplot 3: Area Worst vs Concave Points Worst
    axes[2].scatter(benign['worst area'], benign['worst concave points'], color='#2b5c8f', alpha=0.7, label='Benign', s=28)
    axes[2].scatter(malignant['worst area'], malignant['worst concave points'], color='#d9534f', alpha=0.8, label='Malignant', s=32, marker='^')
    axes[2].set_title("Worst Area vs Worst Concave Points", fontsize=13, fontweight='bold', pad=12)
    axes[2].set_xlabel("Worst Area", fontsize=11)
    axes[2].set_ylabel("Worst Concave Points", fontsize=11)
    axes[2].legend(frameon=True, facecolor='white', loc='lower right')
    axes[2].grid(linestyle='--', alpha=0.5)
    
    plt.savefig('assets/fig1_data_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved assets/fig1_data_distribution.png")

# 2. Figure 2: Architecture Overview Schematic
def plot_architectures():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), dpi=300)
    
    models_info = [
        ("1. Baseline MLP", ["Input (30 features)", "Dense(64, ReLU)", "Dense(1, Sigmoid)"], "#4A90E2"),
        ("2. Deep MLP", ["Input (30 features)", "Dense(128, ReLU)", "Dense(64, ReLU)", "Dense(32, ReLU)", "Dense(1, Sigmoid)"], "#50E3C2"),
        ("3. Regularized MLP", ["Input (30 features)", "Dense(128) + BN + ReLU\n+ Dropout(0.3) + L2", "Dense(64) + BN + ReLU\n+ Dropout(0.2) + L2", "Dense(32) + BN + ReLU\n+ Dropout(0.1) + L2", "Dense(1, Sigmoid)"], "#F5A623"),
        ("4. Residual MLP (ResMLP)", ["Input (30 features)", "Dense Proj(64) + LN", "Residual Block 1\n[Dense-GELU-Drop-Dense]\n+ Skip Add + LN", "Residual Block 2\n[Dense-GELU-Drop-Dense]\n+ Skip Add + LN", "Head: Dense(32) + Drop\nDense(1, Sigmoid)"], "#BD10E0")
    ]
    
    for ax, (title, layers_list, color) in zip(axes, models_info):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10, color='#1A252C')
        
        n = len(layers_list)
        box_height = 1.3
        spacing = 9.5 / n
        
        for i, text in enumerate(layers_list):
            y = 10.5 - (i * spacing) - box_height / 2
            rect = patches.FancyBboxPatch(
                (1, y), 8, box_height, boxstyle="round,pad=0.2",
                facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=0.85
            )
            ax.add_patch(rect)
            ax.text(5, y + box_height/2, text, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color='white' if color != '#50E3C2' else '#1A252C')
            
            if i < n - 1:
                ax.annotate('', xy=(5, y), xytext=(5, y - (spacing - box_height)),
                            arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5))
                            
    plt.savefig('assets/fig2_architectures.png', bbox_inches='tight')
    plt.close()
    print("Saved assets/fig2_architectures.png")

# 3. Figure 3: 5-Fold Cross Validation Results
def plot_kfold_results():
    with open('results/kfold_results.json') as f:
        kfold_data = json.load(f)
        
    metrics = ['accuracy', 'recall', 'specificity', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Recall (Sens)', 'Specificity', 'F1-Score', 'ROC-AUC']
    models = ['Baseline_MLP', 'Deep_MLP', 'Regularized_MLP', 'Residual_MLP']
    labels = ['Baseline MLP', 'Deep MLP', 'Reg MLP', 'Residual MLP']
    colors = ['#2b5c8f', '#3690c0', '#67a9cf', '#02818a']
    
    x = np.arange(len(metrics))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    for i, (m, lbl) in enumerate(zip(models, labels)):
        means = [kfold_data[m]['summary'][metric + '_mean'] * 100 for metric in metrics]
        stds = [kfold_data[m]['summary'][metric + '_std'] * 100 for metric in metrics]
        pos = x + (i - 1.5) * width
        rects = ax.bar(pos, means, width, yerr=stds, capsize=3, label=lbl, color=colors[i], edgecolor='black', alpha=0.9)
        
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Perbandingan Kinerja 5-Fold Stratified Cross-Validation (Mean ± Std)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(92, 101)
    ax.legend(frameon=True, loc='lower right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig('assets/fig3_kfold_cv_results.png', bbox_inches='tight')
    plt.close()
    print("Saved assets/fig3_kfold_cv_results.png")

# 4. Figure 4: Learning Curves
def plot_learning_curves():
    with open('results/training_histories.json') as f:
        hists = json.load(f)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    colors = {'Baseline_MLP': '#2b5c8f', 'Deep_MLP': '#e7298a', 'Regularized_MLP': '#e6ab02', 'Residual_MLP': '#1b9e77'}
    
    for name, data in hists.items():
        epochs = range(1, len(data['loss']) + 1)
        axes[0].plot(epochs, data['loss'], label=f"{name} (Train)", color=colors[name], linestyle='--', alpha=0.6)
        axes[0].plot(epochs, data['val_loss'], label=f"{name} (Val)", color=colors[name], linewidth=2.0)
        
        axes[1].plot(epochs, data['accuracy'], label=f"{name} (Train)", color=colors[name], linestyle='--', alpha=0.6)
        axes[1].plot(epochs, data['val_accuracy'], label=f"{name} (Val)", color=colors[name], linewidth=2.0)
        
    axes[0].set_title("Training & Validation Loss Convergence", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Loss (Binary Crossentropy)", fontsize=11)
    axes[0].grid(linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=8.5, ncol=2)
    
    axes[1].set_title("Training & Validation Accuracy", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].set_ylim(0.85, 1.01)
    axes[1].grid(linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=8.5, ncol=2)
    
    plt.savefig('assets/fig4_learning_curves.png', bbox_inches='tight')
    plt.close()
    print("Saved assets/fig4_learning_curves.png")

# 5. Figure 5: Optimizer & Learning Rate Tuning
def plot_tuning_results():
    with open('results/optimizer_results.json') as f:
        opt_data = json.load(f)
    with open('results/lr_results.json') as f:
        lr_data = json.load(f)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Optimizer loss
    opt_colors = {'adam': '#2b5c8f', 'rmsprop': '#d95f02', 'sgd': '#7570b3'}
    for opt, vals in opt_data.items():
        axes[0].plot(vals['val_loss'], label=f"{opt.upper()} (Val Loss)", color=opt_colors[opt], lw=2)
    axes[0].set_title("Perbandingan Optimizer pada Regularized MLP", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Validation Loss", fontsize=11)
    axes[0].set_ylim(0, 0.6)
    axes[0].legend(fontsize=10)
    axes[0].grid(linestyle='--', alpha=0.5)
    
    # Learning rate loss
    lr_colors = {'0.01': '#e41a1c', '0.001': '#377eb8', '0.0001': '#4daf4a'}
    for lr, vals in lr_data.items():
        axes[1].plot(vals['val_loss'], label=f"LR = {lr} (Val Loss)", color=lr_colors[lr], lw=2)
    axes[1].set_title("Tuning Learning Rate pada Residual MLP", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Validation Loss", fontsize=11)
    axes[1].set_ylim(0, 0.6)
    axes[1].legend(fontsize=10)
    axes[1].grid(linestyle='--', alpha=0.5)
    
    plt.savefig('assets/fig5_optimizer_tuning.png', bbox_inches='tight')
    plt.close()
    print("Saved assets/fig5_optimizer_tuning.png")

# 6. Figure 6: Test Set Confusion Matrices
def plot_confusion_matrices():
    with open('results/test_results.json') as f:
        test_data = json.load(f)
        
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=300)
    names = ['Baseline_MLP', 'Deep_MLP', 'Regularized_MLP', 'Residual_MLP']
    titles = ['Baseline MLP', 'Deep MLP', 'Regularized MLP', 'Residual MLP']
    
    for ax, name, title in zip(axes, names, titles):
        m = test_data[name]['metrics']
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0, vmax=75)
        ax.set_title(f"{title}\nAcc: {m['accuracy']*100:.1f}%, Rec: {m['recall']*100:.1f}%", fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Benign (0)', 'Malignant (1)'], fontsize=10)
        ax.set_yticklabels(['Benign (0)', 'Malignant (1)'], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight='bold')
        ax.set_ylabel("Actual Label", fontsize=10, fontweight='bold')
        
        # Annotate text
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                tag = "(TN)" if (i==0 and j==0) else ("(FP)" if (i==0 and j==1) else ("(FN)" if (i==1 and j==0) else "(TP)"))
                color = 'white' if val > 35 else 'black'
                ax.text(j, i, f"{val}\n{tag}", ha="center", va="center", color=color, fontweight='bold', fontsize=11)
                
    plt.savefig('assets/fig6_test_confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("Saved assets/fig6_test_confusion_matrix.png")

# 7. Figure 7: ROC Curves
def plot_roc_curves():
    with open('results/test_results.json') as f:
        test_data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = {'Baseline_MLP': '#2b5c8f', 'Deep_MLP': '#e7298a', 'Regularized_MLP': '#d95f02', 'Residual_MLP': '#1b9e77'}
    labels = {'Baseline_MLP': 'Baseline MLP', 'Deep_MLP': 'Deep MLP', 'Regularized_MLP': 'Regularized MLP', 'Residual_MLP': 'Residual MLP'}
    
    for name in colors.keys():
        fpr = test_data[name]['fpr']
        tpr = test_data[name]['tpr']
        auc = test_data[name]['metrics']['roc_auc']
        ax.plot(fpr, tpr, label=f"{labels[name]} (AUC = {auc:.4f})", color=colors[name], lw=2.5)
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.5000)')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Kurva pada Test Set', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.grid(linestyle='--', alpha=0.5)
    
    plt.savefig('assets/fig7_roc_curves.png', bbox_inches='tight')
    plt.close()
    print("Saved assets/fig7_roc_curves.png")

if __name__ == '__main__':
    plot_data_distribution()
    plot_architectures()
    plot_kfold_results()
    plot_learning_curves()
    plot_tuning_results()
    plot_confusion_matrices()
    plot_roc_curves()
    print("All 7 figures generated successfully!")
