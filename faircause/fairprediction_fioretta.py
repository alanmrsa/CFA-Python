from faircause.utils.prediction_helpers import *
from faircause.utils.neural_learning import *
from faircause.faircause import *
from sklearn.model_selection import train_test_split
from faircause.utils.prediction_generics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

class FairPredict_Fioretta: 
    '''
    data: pandas.DataFrame
        The data to train the model on
    X: str
        The name of the protected attribute
    Z: list
        The names of the confounders  
    W: list
        The names of the covariates
    Y: str
        The name of the outcome variable
    x0: Any
        The reference level of the protected attribute
    x1: Any
        The comparison level of the protected attribute
    BN: list
        The business necessity set
    eval_prop: float
        The proportion of the data to use for evaluation
    lr: float
        The learning rate
    step_sizes: list
        The step sizes for the Fioretto method
    relu_eps: bool
        Whether to use the ReLU epsilon method
    patience: int
        The number of epochs to wait before stopping the training process
    method: str
        The method to use for the causal estimation, either "debiasing" or "medDML"
    model: str
        The model to use for the causal estimation, either "ranger" or "linear"
    tune_params: bool
        Whether to tune the parameters for the causal estimation
    nboot: int
        The number of bootstraps to use for the causal estimation
    nboot2: int
        The number of bootstraps to use for the causal estimation
    random_seed: int
        The random seed to use 
    kwargs: dict
        Additional keyword arguments

    Methods
    fioretta_train()
        Train using the Fioretto method
    predict()
        Predict the outcome for new data
    plot_training()
        Plot the training history
    '''
    def __init__(self, data, X, Z, W, Y, x0, x1, BN='',
                     eval_prop=0.25, lr=0.001, 
                     step_sizes=[0.0001],
                     relu_eps=False, patience=100,
                     method="medDML", model="ranger",
                     tune_params=False, nboot=1, 
                     nboot2=100, random_seed = None, **kwargs):
        
        if random_seed is not None: 
            np.random.seed(random_seed)
        self.random_seed = random_seed

        self.data = data   
        self.X = X
        self.Z = Z
        self.W = W
        self.Y = Y
        self.x0 = x0
        self.x1 = x1
        self.BN = BN
        self.eval_prop = eval_prop
        self.lr = lr
        self.step_sizes = step_sizes
        self.relu_eps = relu_eps
        self.patience = patience
        self.method = method
        self.model = model
        self.tune_params = tune_params
        self.nboot = nboot
        self.nboot2 = nboot2
        self.kwargs = kwargs
        self.y_meas = None 
        self.yhat_meas = None
        self.training_hist = None
        self.nn_mod = None

        # Validate inputs
        self.verify_numeric_input(data)
        
        self.y_fcb = FairCause(data, X=X, Z=Z, W=W, Y=Y, x0=x0, x1=x1,
                      model=model, method=method,
                      tune_params=tune_params, n_boot1=nboot, n_boot2=nboot2, random_seed=random_seed) 
        
    
    def verify_numeric_input(self, data):    
        if isinstance(data, pd.DataFrame):
            non_numeric_cols = [col for col in data.columns 
                            if not pd.api.types.is_numeric_dtype(data[col])]
            
            if non_numeric_cols:
                raise ValueError(
                    "Only `numeric` and `integer` values currently supported as inputs "
                    "for `fair_prediction()` functionality.\n"
                    "If you have `factors` or `logicals`, please convert them using "
                    "one-hot encoding."
                )
        else: 
            raise ValueError(
                "Please pass a `pandas.DataFrame` to `fair_prediction()`"
            )

    def fioretta_train(self) :

        self.y_fcb.estimate_effects()  

        if self.random_seed is not None: 
            train_data, eval_data = train_test_split(self.data, test_size=self.eval_prop, random_state=self.random_seed)
        else: 
            train_data, eval_data = train_test_split(self.data, test_size=self.eval_prop)
        y_meas = self.y_fcb.summary(decompose="general")
        task_type = "regression" if len(self.data[self.Y].unique()) > 2 else "classification"

        best_model_global, loss_hist, nde_hist, nie_hist, nse0_hist, nse1_hist, nde_lmbd_hist, nie_lmbd_hist, nse0_lmbd_hist, nse1_lmbd_hist = train_w_fioretta(
            train_data, eval_data, x_col=self.X, w_cols=self.W, z_cols=self.Z, 
            y_col=self.Y, step_sizes= self.step_sizes, lr=self.lr, 
            nde="DE" not in self.BN, 
            nie="IE" not in self.BN,
            nse="SE" not in self.BN,
            eta_de=y_meas.loc[y_meas['measure'] == 'nde', 'value'].iloc[0],
            eta_ie=y_meas.loc[y_meas['measure'] == 'nie', 'value'].iloc[0],
            eta_se_x0=y_meas.loc[y_meas['measure'] == 'expse_x0', 'value'].iloc[0],
            eta_se_x1=y_meas.loc[y_meas['measure'] == 'expse_x1', 'value'].iloc[0],
            verbose=False,
            relu_eps=self.relu_eps,
            patience=self.patience,
            seed=self.random_seed
        )

        self.nn_mod = best_model_global
        self.training_metrics = {
            'loss_history': loss_hist,
            'nde_history': nde_hist,
            'nie_history': nie_hist,
            'nse0_history': nse0_hist,
            'nse1_history': nse1_hist,
            'nde_lmbd_history': nde_lmbd_hist,
            'nie_lmbd_history': nie_lmbd_hist,
            'nse0_lmbd_history': nse0_lmbd_hist,
            'nse1_lmbd_history': nse1_lmbd_hist
        }

        eval_data['preds'] = None
        tmp = [self.X] + self.Z+ self.W
        eval_data['preds'] = pred_nn_proba(self.nn_mod, eval_data[tmp], task_type)

        eval_data = eval_data.reset_index(drop=True)
        eval_fcb = FairCause(eval_data, X=self.X, Z=self.Z, W=self.W, Y="preds",
                                x0=self.x0, x1=self.x1,
                                model=self.model, method=self.method,
                                tune_params=self.tune_params, n_boot1=self.nboot, n_boot2=self.nboot2, random_seed=self.random_seed)
        
        eval_fcb.estimate_effects()
        y_eval = eval_data[self.Y]
        p_eval = eval_data["preds"]
        meas=eval_fcb.summary(decompose="general")

        self.yhat_meas = fioretta_performance(meas, y_eval, p_eval)
        self.y_meas = y_meas
        
    
    def predict(self, newdata):
        """
        Predict outcomes for new data using the trained Fioretta model.
        """
        if self.nn_mod is None:
            raise ValueError("No trained model found. Run fioretta_train() first.")

        # Prepare features
        tmp = [self.X] + self.Z + self.W
        features = newdata[tmp]
        X_tensor = torch.tensor(features.values, dtype=torch.float)
        model = self.nn_mod

        # Predict
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor)
            probs = torch.sigmoid(predictions).numpy().flatten()

        newdata = newdata.copy()
        newdata['preds'] = probs

        # Evaluate causal effects on predictions
        test_fcb = FairCause(
            newdata, X=self.X, Z=self.Z, W=self.W, Y="preds",
            x0=self.x0, x1=self.x1,
            model=self.model, method=self.method,
            tune_params=self.tune_params, n_boot1=self.nboot, n_boot2=self.nboot2, random_seed=self.random_seed
        )
        test_fcb.estimate_effects()
        y_test = newdata[self.Y]
        p_test = newdata["preds"]
        meas = test_fcb.summary(decompose="general")

        result = {
            'predictions': probs.tolist(),
            'test_meas': meas,
            'y_meas': self.y_meas,
            'BN': self.BN,
        }
        return result

    def plot_training(self, figsize=(15, 10)):
        if self.training_metrics is None:
            raise ValueError("No training metrics available. Run fioretta_train() first.")
            
        return plot_fioretta_training(self.training_metrics, figsize=figsize)