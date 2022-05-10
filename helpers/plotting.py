import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
np.random.seed(1)

def plot_p_vals(p_vals_baseline, p_vals_sloe, title, filename, n, ratio):
    fig, ax = plt.subplots(1, 2, sharey='row', figsize=(10,5))
    
    ax[0].hist(p_vals_baseline, bins=20, density=True)
    ax[0].axhline(y=1.0, color='black', linestyle='--')
    ax[0].set_xlabel('p-values')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Standard')

    ax[1].hist(p_vals_sloe, bins=20, density=True)
    ax[1].axhline(y=1.0, color='black', linestyle='--')
    ax[1].set_xlabel('p-values')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Corrected')
    
    fig.suptitle('{} p-values (n={}, ratio={})'.format(title, n, ratio))
    plt.tight_layout()
    plt.savefig('results/plots/{}/{}_{}_{}_p_vals.pdf'.format(filename, filename, n, ratio))
    plt.savefig('results/plots/{}/{}_{}_{}_p_vals.jpg'.format(filename, filename, n, ratio))
    plt.clf()

def plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test, y_test, title, filename, n, ratio, ci=90):
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    #Sample from test set for CI
    num_samples = 8
    sample_idxs = np.random.randint(0, X_test.shape[0], size=(num_samples,))

    # x axis bounds for logit plot
    min_xval = 1e-8
    max_xval = 1 - 1e-8

    spacing = 0.1

    for i, row_idx in enumerate(sample_idxs):
        # Observed y value
        observed = y_test[row_idx]
        if observed == 0.0:
            observed_logit = min_xval
        else:
            observed_logit = max_xval

        # Baseline CI
        lower_ci_baseline = pred_ints_baseline[row_idx,0]
        upper_ci_baseline = pred_ints_baseline[row_idx,1]
        # SLOE CI
        lower_ci_sloe = pred_ints_sloe[row_idx,0]
        upper_ci_sloe = pred_ints_sloe[row_idx,1]
        # Set values for logit
        lower_ci_baseline_logit = min(max(min_xval, lower_ci_baseline), max_xval)
        upper_ci_baseline_logit = max(min(max_xval, upper_ci_baseline), min_xval)
        lower_ci_sloe_logit = min(max(min_xval, lower_ci_sloe), max_xval)
        upper_ci_sloe_logit = max(min(max_xval, upper_ci_sloe), min_xval)
            
        ax[0].plot((lower_ci_baseline,upper_ci_baseline),(i-spacing,i-spacing),'b|-',color='blue', linewidth=2)
        ax[1].plot((lower_ci_baseline_logit,upper_ci_baseline_logit),(i-spacing,i-spacing),'b|-',color='blue', linewidth=2)

        ax[0].plot((lower_ci_sloe,upper_ci_sloe),(i+spacing,i+spacing),'r|-',color='orange', linewidth=2)
        ax[1].plot((lower_ci_sloe_logit,upper_ci_sloe_logit),(i+spacing,i+spacing),'r|-',color='orange', linewidth=2)

        ax[0].plot(observed, i, '*', color='black')
        ax[1].plot(observed_logit, i, '*', color='black')
        
    ax[0].set_yticks([])
    ax[0].set_ylabel('Example')
    ax[0].set_xlabel('Probability')
    ax[1].set_yticks([])
    ax[1].set_ylabel('Example')
    ax[1].set_xlabel('Probability (logit scale)')
    ax[1].set_xscale("logit")
    
    fig.suptitle('{} Confidence Intervals (n={}, ratio={})'.format(title, n, ratio))
    observed_patch = mlines.Line2D([],[],color='black', marker='*', linestyle='None', label='Observed Outcome')
    standard_patch = mlines.Line2D([],[],color='blue', marker='|',label='Standard {}% CI'.format(ci))
    corrected_patch = mlines.Line2D([],[],color='orange', marker='|',label='Corrected {}% CI'.format(ci))
    ax[1].legend(handles=[observed_patch, standard_patch, corrected_patch])
    plt.tight_layout()

    plt.savefig('results/plots/{}/{}_{}_{}_CIs.pdf'.format(filename, filename, n, ratio))
    plt.savefig('results/plots/{}/{}_{}_{}_CIs.jpg'.format(filename, filename, n, ratio))
    plt.clf()