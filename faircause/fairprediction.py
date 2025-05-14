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
import scipy.stats as stats

class FairPredict: 
    """
    FairPredict is a class that implements the FairCause framework for fair prediction.
    It is used to estimate the effects of treatment on the outcome, while controlling for 
    the effects of confounders and selection bias.

    Parameters:
    data: pandas.DataFrame
        The data to be used for the fair prediction.
    X: str
        The name of the treatment variable.
    Z: list of str
        The names of the confounders.
    W: list of str
        The names of the mediators . 
    Y: str
        The name of the outcome variable.
    x0: float
        The value of the treatment variable for the control group.
    x1: float
        The value of the treatment variable for the treatment group.
    BN: list of str
        The business necessity set.
    eval_prop: float
        The proportion of the data to be used for the evaluation set.
    lr: float
        The learning rate for the model.
    lmbd_seq: list of float
        The sequence of lambda values to be used for the model.
    relu_eps: bool
        Whether to use the relu epsilon trick.
    patience: int
        The number of epochs to wait before stopping the training of the model.
    method: str
        Either "debiased" or "medDML". The method to be used for the model.
    model: str
        Either "ranger" or "linear". The model to be used FairCause causal estimation.
    tune_params: bool
        Whether to tune the parameters of the model in FairCause for estimation.
    nboot: int
        The number of bootstrap samples to be used for estimation.
    nboot2: int
        The number of inner bootstrap samples to be used for estimation.
    random_seed: int
        The random seed to be used for the model.

    Methods
    train()
        Train the model
    predict()
        Predict the outcome for new data
    plot()
        Plot the fairness measures decomposition
    """
    def __init__(self, data, X, Z, W, Y, x0, x1, BN='',
                     eval_prop=0.25, lr=0.001, 
                     lmbd_seq=[0.1,0.5,1,2,5,10],
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
        self.lmbd_seq = lmbd_seq
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
    
    def train(self) :

        self.y_fcb.estimate_effects()  
        if self.random_seed is not None: 
            train_data, eval_data = train_test_split(self.data, test_size=self.eval_prop, random_state=self.random_seed)
        else: 
            train_data, eval_data = train_test_split(self.data, test_size=self.eval_prop)

        y_meas = self.y_fcb.summary(decompose="general")
        
        nn_mod = {lmbd: None for lmbd in self.lmbd_seq}

        task_type = "regression" if len(self.data[self.Y].unique()) > 2 else "classification"
        res = []
        for lmbd in self.lmbd_seq:
            nn_mod[lmbd] = train_w_es(
                train_data, eval_data, x_col=self.X, w_cols=self.W, z_cols=self.Z, 
                y_col=self.Y, lmbd=lmbd, lr=self.lr, 
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
        
            current_model = nn_mod[lmbd]
            eval_data['preds'] = None
            tmp = [self.X] + self.Z+ self.W
            eval_data['preds'] = pred_nn_proba(current_model, eval_data[tmp], task_type)

            eval_data = eval_data.reset_index(drop=True)
            eval_fcb = FairCause(eval_data, X=self.X, Z=self.Z, W=self.W, Y="preds",
                                 x0=self.x0, x1=self.x1,
                                 model=self.model, method=self.method,
                                 tune_params=self.tune_params, n_boot1=self.nboot, n_boot2=self.nboot2, random_seed=self.random_seed)

            eval_fcb.estimate_effects()
            y_eval = eval_data[self.Y]
            p_eval = eval_data["preds"]
            meas=eval_fcb.summary(decompose="general")
            res.append(lambda_performance(meas, y_eval, p_eval, lmbd))

        self.y_meas = y_meas
        self.yhat_meas = pd.concat(res)
        self.nn_mod = nn_mod

    def binary_train(self, lmbd = 10, stop_eps = 0.1): 
        lambda_low = 0
        if self.random_seed is not None: 
            train_data, eval_data = train_test_split(self.data, test_size=self.eval_prop, random_state=self.random_seed)
        else: 
            train_data, eval_data = train_test_split(self.data, test_size=self.eval_prop)
        task_type = "regression" if len(self.data[self.Y].unique()) > 2 else "classification"

        self.y_fcb.estimate_effects()  
        self.y_meas = self.y_fcb.summary(decompose="general")
        lambda_high = lmbd

        while np.abs(lambda_high - lambda_low) > stop_eps: 
            lambda_mid = (lambda_high + lambda_low) / 2
            model = train_w_es(train_data, eval_data, x_col=self.X, w_cols=self.W, z_cols=self.Z, 
                y_col=self.Y, lmbd=lambda_mid, lr=self.lr, 
                nde="DE" not in self.BN, nie="IE" not in self.BN, nse="SE" not in self.BN,
                verbose=False, relu_eps=self.relu_eps, patience=self.patience, seed=self.random_seed)
            
            eval_data['preds'] = None
            tmp = [self.X] + self.Z+ self.W
            eval_data['preds'] = pred_nn_proba(model, eval_data[tmp], task_type)

            eval_data = eval_data.reset_index(drop=True)
            eval_fcb = FairCause(eval_data, X=self.X, Z=self.Z, W=self.W, Y="preds",
                                 x0=self.x0, x1=self.x1,
                                 model=self.model, method=self.method,
                                 tune_params=self.tune_params, n_boot1=self.nboot, n_boot2=self.nboot2, random_seed=self.random_seed)

            eval_fcb.estimate_effects()
            y_eval = eval_data[self.Y]
            p_eval = eval_data["preds"]
            meas=eval_fcb.summary(decompose="general")
            result = lambda_performance(meas, y_eval, p_eval, lambda_mid)


            flag = False
            if (hypo_test(result.loc[result['measure'] == 'nde', 'value'].iloc[0], result.loc[result['measure'] == 'nde', 'sd'].iloc[0],0, 0)  if 'DE' not in self.BN else 
                hypo_test(result.loc[result['measure'] == 'nde', 'value'].iloc[0], result.loc[result['measure'] == 'nde', 'sd'].iloc[0],meas.loc[meas['measure'] == 'nde', 'value'].iloc[0], meas.loc[meas['measure'] == 'nde', 'sd'].iloc[0])): 
                flag = True
            
            if (hypo_test(result.loc[result['measure'] == 'nie', 'value'].iloc[0], result.loc[result['measure'] == 'nie', 'sd'].iloc[0],0, 0)  if 'IE' not in self.BN else 
                hypo_test(result.loc[result['measure'] == 'nie', 'value'].iloc[0], result.loc[result['measure'] == 'nie', 'sd'].iloc[0],meas.loc[meas['measure'] == 'nie', 'value'].iloc[0], meas.loc[meas['measure'] == 'nie', 'sd'].iloc[0])): 
                flag = True
            
            if (hypo_test(result.loc[result['measure'] == 'expse_x0', 'value'].iloc[0], result.loc[result['measure'] == 'expse_x0', 'sd'].iloc[0],0, 0)  if 'SE' not in self.BN else 
                hypo_test(result.loc[result['measure'] == 'expse_x0', 'value'].iloc[0], result.loc[result['measure'] == 'expse_x0', 'sd'].iloc[0],meas.loc[meas['measure'] == 'expse_x0', 'value'].iloc[0], meas.loc[meas['measure'] == 'expse_x0', 'sd'].iloc[0])): 
                flag = True
            
            if flag: 
                lambda_low = lambda_mid
            else: 
                lambda_high = lambda_mid
        if self.nn_mod is None: 
            self.nn_mod = {}
        self.nn_mod[lambda_mid] = model
        return (model, lambda_mid)
                
    
    def predict(self, newdata): 
        test_meas = None
        preds = {lmbd: None for lmbd in self.nn_mod.keys()}
        y_meas = self.y_meas

        if len(self.data[self.Y].unique()) > 2:
            raise ValueError("Only implemented for binary classification")

        for lmbd in self.nn_mod.keys():
            tmp = [self.X] + self.Z+ self.W
            features = newdata[tmp]
            X_tensor = torch.tensor(features.values, dtype=torch.float)
            model = self.nn_mod[lmbd]

            model.eval()
            with torch.no_grad(): 
                predictions = model(X_tensor)

                #only setup for binary classification as of now: implement later
                probs = torch.sigmoid(predictions).numpy().flatten()  

            newdata['preds'] = probs
            preds[lmbd] = probs.tolist()

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

            lambda_perf = lambda_performance(meas, y_test, p_test, lmbd)

            if test_meas is None: 
                test_meas = pd.DataFrame(lambda_perf.squeeze())
            else: 
                test_meas = pd.concat([test_meas, pd.DataFrame(lambda_perf).squeeze()], axis=0, ignore_index=True)

        result = {
            'predictions': preds, 
            'test_meas': test_meas, 
            'y_meas': y_meas, 
            'BN': self.BN, 
        }   

        return result
    def plot(self, type): 
        autoplot_fair_prediction(self.yhat_meas, self.y_meas, self.BN, type)