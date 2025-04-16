import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def format_measures(measures, method='medDML'):
    """Helper function to summarize measures similar to the R summarize_measures function"""
    if method == "debiasing":
        if 'scale' in measures:
            measures = measures.drop(columns=['scale'])
        return measures
    
    # Group by measure and calculate mean and std
    summary = measures.groupby('measure').agg({
        'value': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-index columns
    summary.columns = ['measure', 'value', 'sd']
    return summary

def print_summary(X, Z, W, Y, x0, x1, measures, decompose="xspec"):
    """Print a summary of fairness measures"""
    # Get measures as a dictionary for easy access
    meas_dict = {row['measure']: row['value'] for _, row in measures.iterrows()}
    
    # Create header
    output = [
        "faircause object summary:",
        f"\nProtected attribute:                 {X}",
        f"Protected attribute levels:          {x0}, {x1}",
        f"Total Variation (TV): {meas_dict['tv']:.4f}",
        "\nTV decomposition(s):"
    ]
    
    # Add decompositions
    if decompose in ["general", "both"]:
        output.append(
            f"\nTV_{x0}{x1}(y) ({meas_dict['tv']:.4f}) = "
            f"NDE_{x0}{x1}(y) ({meas_dict['nde']:.4f}) - "
            f"NIE_{x1}{x0}(y) ({meas_dict['nie']:.4f}) + "
            f"ExpSE_{x1}(y) ({meas_dict['expse_x1']:.4f}) - "
            f"ExpSE_{x0}(y) ({meas_dict['expse_x0']:.4f})"
        )
        
    if decompose in ["xspec", "both"]:
        output.append(
            f"\nTV_{x0}{x1}(y) ({meas_dict['tv']:.4f}) = "
            f"CtfDE_{x0}{x1}(y | {x0}) ({meas_dict['ctfde']:.4f}) - "
            f"CtfIE_{x1}{x0}(y | {x0}) ({meas_dict['ctfie']:.4f}) - "
            f"CtfSE_{x1}{x0}(y) ({meas_dict['ctfse']:.4f})"
        )
        
    return "\n".join(output)

def create_fairness_plot(X, Z, W, Y, x0, x1, measures, decompose="xspec", dataset="", signed=True, var_name="y"):
    """Create a plot of fairness measures"""
    # Rename columns to match R version
    df = measures.rename(columns={'measure': 'Measure', 'value': 'Value', 'sd': 'StdDev'})
    
    # Define measure labels with LaTeX format
    rename = {
        'tv': f"$TV_{{{x0}, {x1}}}({var_name})$",
        'te': f"$TE_{{{x0}, {x1}}}({var_name})$",
        'expse_x1': f"$Exp$-$SE_{{{x1}}}({var_name})$",
        'expse_x0': f"$Exp$-$SE_{{{x0}}}({var_name})$",
        'nde': f"$NDE_{{{x0}, {x1}}}({var_name})$",
        'nie': f"$NIE_{{{x1}, {x0}}}({var_name})$",
        'ett': f"$ETT_{{{x0}, {x1}}}({var_name} | {x0})$",
        'ctfde': f"$Ctf$-$DE_{{{x0}, {x1}}}({var_name} | {x0})$",
        'ctfie': f"$Ctf$-$IE_{{{x1}, {x0}}}({var_name} | {x0})$",
        'ctfse': f"$Ctf$-$SE_{{{x1}, {x0}}}({var_name})$"
    }
    
    # Set title
    title = f"${rename['tv']}$ decomposition {dataset}"
    
    # Handle signed option
    if not signed:
        assert decompose == "xspec", "Signed=False only supported for decompose='xspec'"
        rename['tv'] = f"$PG_{{{x0}, {x1}}}({var_name})$"
        title = f"${rename['tv']}$ decomposition {dataset}"
        
        # Flip signs of ctfie and ctfse
        ctfie_idx = df['Measure'] == 'ctfie'
        ctfse_idx = df['Measure'] == 'ctfse'
        df.loc[ctfie_idx | ctfse_idx, 'Value'] *= -1
    
    # Include measures based on decomposition type
    if decompose == "xspec":
        inc_meas = ['tv', 'ctfde', 'ctfie', 'ctfse']
    elif decompose == "general":
        inc_meas = ['tv', 'nde', 'nie', 'expse_x0', 'expse_x1']
    else:  # both
        inc_meas = list(rename.keys())
    
    # Filter measures
    df = df[df['Measure'].isin(inc_meas)]
    
    # Create categorical type for proper ordering
    measure_order = [m for m in list(rename.keys()) if m in inc_meas]
    df['Measure'] = pd.Categorical(df['Measure'], categories=measure_order)
    df = df.sort_values('Measure')
    
    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Measure', y='Value', data=df, palette='Set2')
    
    # Add error bars
    plt.errorbar(
        x=range(len(df)),
        y=df['Value'],
        yerr=1.96 * df['StdDev'],
        fmt='none',
        color='black',
        capsize=5
    )
    
    # Set x-axis labels
    ax.set_xticklabels([rename.get(m, m) for m in df['Measure']], rotation=45, ha='right')
    
    # Set labels and title
    plt.xlabel('Causal Fairness Measure', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

