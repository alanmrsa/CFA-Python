import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def lambda_solution_path(yhat_meas, y_meas, BN): 
    def effect_path(res, gt, meas, bn, ax): 
        # Added 'ax' parameter to use the provided axes
        res_filtered = res[res['measure'] == meas]
        gt_filtered = gt[gt['measure'] == meas]

        if bn: 
            yintercept = gt_filtered['value'].iloc[0]
            clr = 'darkgreen'
        else: 
            yintercept = 0
            clr = 'darkred'

        # Use the provided axis instead of creating a new one
        ax.plot(res_filtered['lmbd'], res_filtered['value'], marker='o')
        ax.fill_between(res_filtered['lmbd'], res_filtered['value'] - res_filtered['sd'], 
                        res_filtered['value'] + res_filtered['sd'], alpha=0.2)
        
        ax.scatter(gt_filtered['lmbd'], gt_filtered['value'], color='blue')
        ax.errorbar(gt_filtered['lmbd'], gt_filtered['value'],
            yerr=gt_filtered['sd'], color='blue',
            capsize=5, fmt='o')
        ax.axhline(yintercept, color=clr, linestyle='-')
        ax.set_title(meas.upper(), color=clr)
        ax.grid(True, alpha=0.3)
        # Remove plt.close(fig) as we're using the provided axes
        return ax

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Create each subplot
    effect_path(yhat_meas, y_meas, "nde", "DE" in BN, axes[0])
    effect_path(yhat_meas, y_meas, "nie", "IE" in BN, axes[1])
    effect_path(yhat_meas, y_meas, "expse_x1", "SE" in BN, axes[2])
    effect_path(yhat_meas, y_meas, "expse_x0", "SE" in BN, axes[3])
    
    plt.tight_layout()
    return fig

def autoplot_fair_prediction(yhat_meas, y_meas, BN, type, task_type="classification"): 
    if type not in ["causal", "accuracy"]:
        raise ValueError("type must be either 'causal' or 'accuracy'")
    
    eval_meas = yhat_meas
    y_meas = y_meas
    
    # Add lambda column to y_meas
    y_meas['lmbd'] = -0.5
    
    if type == "causal":
        # Call the lambda_solution_path function which would need to be implemented
        return lambda_solution_path(eval_meas, y_meas, BN)
    
    elif type == "accuracy":
        # implement reg later when we figure it out
        if task_type == "classification":
            fig = plt.figure(figsize=(15, 5))
            gs = GridSpec(1, 3, figure=fig)
            
            # Cross-entropy plot
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(eval_meas['lmbd'], eval_meas['bce'], marker='o')
            ax1.fill_between(eval_meas['lmbd'], 
                            eval_meas['bce'] - eval_meas['bce_sd'],
                            eval_meas['bce'] + eval_meas['bce_sd'],
                            alpha=0.3)
            ax1.set_title("Cross-entropy")
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(eval_meas['lmbd'], eval_meas['acc'], marker='o')
            ax2.fill_between(eval_meas['lmbd'], 
                            eval_meas['acc'] - eval_meas['acc_sd'],
                            eval_meas['acc'] + eval_meas['acc_sd'],
                            alpha=0.3)
            ax2.set_title("Accuracy")
            ax2.grid(True, alpha=0.3)
            
            # AUC plot
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(eval_meas['lmbd'], eval_meas['auc'], marker='o')
            ax3.fill_between(eval_meas['lmbd'], 
                            eval_meas['auc'] - eval_meas['auc_sd'],
                            eval_meas['auc'] + eval_meas['auc_sd'],
                            alpha=0.3)
            ax3.axhline(y=0.5, color='orange', linestyle='-')
            ax3.set_title("AUC")
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
        return fig

def plot_fioretta_training(training_metrics, figsize=(15, 10)):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(training_metrics['loss_history'], 'b-', label='Total Loss')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Causal Effects
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(training_metrics['nde_history'], 'r-', label='NDE Loss')
    ax2.plot(training_metrics['nie_history'], 'g-', label='NIE Loss')
    ax2.plot(training_metrics['nse0_history'], 'b-', label='NSE0 Loss')
    ax2.plot(training_metrics['nse1_history'], 'y-', label='NSE1 Loss')
    ax2.set_title('Causal Effects')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Effect Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Lagrangian Multipliers
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(training_metrics['nde_lmbd_history'], 'r-', label='NDE λ')
    ax3.plot(training_metrics['nie_lmbd_history'], 'g-', label='NIE λ')
    ax3.plot(training_metrics['nse0_lmbd_history'], 'b-', label='NSE0 λ')
    ax3.plot(training_metrics['nse1_lmbd_history'], 'y-', label='NSE1 λ')
    ax3.set_title('Lagrangian Multipliers')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Multiplier Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    return fig