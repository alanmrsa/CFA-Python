from copy import deepcopy
import pandas as pd
from faircause.estimation.mediation_dml import ci_mdml
from faircause.estimation.one_step_debiased import *
from faircause.utils.generics import *

class FairCause: 
    '''
    data : pd.DataFrame
            Dataset containing all variables
    X : str
        Name of protected attribute column
    Z : List[str]
        Names of mediator columns
    W : List[str]
        Names of confounder columns
    Y : str
        Name of outcome column
    x0 : Any
        Reference level of protected attribute
    x1 : Any
        Comparison level of protected attribute
    method : str, default="medMDL"
        Estimation method: "debiasing" or "medDML"
    model : str, default="ranger"
        Model type for medDML: "ranger" or "linear"
    tune_params : bool, default=False
        Whether to tune model hyperparameters
    n_boot1 : int, default=1
        Number of outer bootstrap repetitions
    n_boot2 : int, default=100
        Number of inner bootstrap repetitions
    '''

    

    def __init__(self, data, X, Z, W, Y, x0, x1, 
                 method='medDML', model='ranger', 
                 tune_params=False, n_boot1=1, n_boot2=100): 

        self.X = X
        self.Z = Z if Z else []
        self.W = W if W else []
        self.Y = Y
        self.x0 = x0
        self.x1 = x1
        self.method = method
        self.model = model
        self.tune_params = tune_params
        self.n_boot1 = n_boot1
        self.n_boot2 = n_boot2
        self.params = {
            'mns_pxz': None,
            'mns_yz': None, 
            'mns_pxzw': None,
            'mns_yzw': None,
            'mns_eyzw': None
        }
        self.res = []

        # Validate inputs
        self._validate_inputs(data)
        
        # Preprocess data
        self.data = self._preprocess_data(data)
    
    def __str__(self):
        return (
            f"faircause object:\n\n"
            f"Attribute:       {self.X}\n"
            f"Outcome:         {self.Y}\n"
            f"Confounders:     {', '.join(self.W) if self.W else ''}\n"
            f"Mediators:       {', '.join(self.Z) if self.Z else ''}"
        )

    def summary(self, decompose="xspec", print_sum=False):
        """Create and print a summary of the fairness measures"""
        if not self.res:
            raise ValueError("No results available. Run estimate_effects() first.")
            
        # Combine all results
        measures_df = pd.concat(self.res)
        
        # Format measures
        summarized_measures = format_measures(measures_df, self.method)
        
        # Print summary
        summary_text = print_summary(
            self.X, self.Z, self.W, self.Y, 
            self.x0, self.x1, summarized_measures, 
            decompose=decompose
        )
        if print_sum: 
            print(summary_text)
                
        # Return the formatted measures for further use if needed
        return summarized_measures
    
    def plot(self, decompose="xspec", dataset="", signed=True, var_name="y"):
        """Plot the fairness measures decomposition"""
        if not self.res:
            raise ValueError("No results available. Run estimate_effects() first.")
            
        # Combine all results
        measures_df = pd.concat(self.res)
        
        # Format measures
        summarized_measures = format_measures(measures_df, self.method)
        
        # Create plot
        fig = create_fairness_plot(
            self.X, self.Z, self.W, self.Y,
            self.x0, self.x1, summarized_measures,
            decompose=decompose, dataset=dataset,
            signed=signed, var_name=var_name
        )
        
        return fig


    def estimate_effects(self): 
        if self.method == "medMDL": 
            for rep in range(1, self.n_boot1 + 1):
                rep_result = ci_mdml(self.data, self.X, self.Z, self.W, self.Y, self.model, rep,
                                nboot=self.n_boot2, tune_params=self.tune_params,
                                params=deepcopy(self.params))
                
                rep_df = rep_result["results"]
                self.params = rep_result["params"]
                
                if rep == 1:
                    pw = rep_result["pw"]
                
                self.res.append(rep_df)
        if self.method == "debiasing":
            data, sfm = preproc_data(self.data, self.X, self.Z, self.W, self.Y)
            X = sfm['X']
            Z = sfm['Z']
            W = sfm['W']
            Y = sfm['Y']
            
            res = one_step_debias(data, X, Z, W, Y)
            print(res)
            res = res[res['measure'].isin(['tv', 'ett', 'ctfde', 'ctfie', 'ctfse'])]
            self.res.append(res)
            pw = res.attrs.get('pw')


    def _validate_inputs(self, data: pd.DataFrame) -> None:
        """Validate input parameters."""
        # Check that all columns exist in the dataframe
        for col in [self.X, self.Y] + self.Z + self.W:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Check protected attribute has the specified levels
        if not set([self.x0, self.x1]).issubset(set(data[self.X].unique())):
            raise ValueError(f"Protected attribute levels {self.x0} and {self.x1} not found in column {self.X}")
        
        # Validate method
        if self.method not in ["debiasing", "causal_forest", "medDML"]:
            raise ValueError(f"Method must be one of: 'debiasing', 'causal_forest', 'medDML'")
        
        # Validate model
        if self.model not in ["ranger", "linear"]:
            raise ValueError(f"Model must be one of: 'ranger', 'linear'")
        
        if (self.model == "linear") and (self.tune_params==True): 
            raise ValueError("Cannot be linear and tune_params=True")

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        
        data_copy[self.X] = data_copy[self.X].map({self.x0: 0, self.x1: 1})
        
        # Convert outcome to numeric if it's categorical
        if isinstance(data_copy[self.Y].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(data_copy[self.Y]):
            data_copy[self.Y] = data_copy[self.Y].astype('category').cat.codes

        # Convert confounders to numeric if they're categorical
        # TODO: eventually implement some type of choice b/w one-hot and ordinal
        for col in self.W + self.Z:
            if isinstance(data_copy[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(data_copy[col]):
                data_copy[col] = data_copy[col].astype('category').cat.codes
        
        return data_copy
