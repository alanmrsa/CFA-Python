import numpy as np
import pandas as pd
import itertools

def gen_data(n_samples): 
    # define the SFM
    def X(U_x, U_xz): 
        return U_x | U_xz
    def Z(U_xz, U_z): 
        return U_xz & U_z
    def W(X, U_w, Z): 
        return (X & U_w )| Z
    def Y(U_y, X, Z, W): 
        return (U_y & X) | (U_y & Z) | (U_y & W)

    #generate U's
    U_x = np.random.binomial(1, 0.4, n_samples)
    U_z = np.random.binomial(1, 0.8, n_samples)
    U_w = np.random.binomial(1, 0.2, n_samples)
    U_y = np.random.binomial(1, 0.3, n_samples)
    U_xz = np.random.binomial(1, 0.4, n_samples)

    #generate data
    x = X(U_x, U_xz)
    z = Z(U_xz, U_z)
    w = W(x, U_w, z)
    y = Y(U_y, x, z, w)

    return pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'W': w})

# computing the true causal effects, compute standard basis expansion for each
# step 1, ID formulas

def X(U_x, U_xz): 
    return U_x | U_xz
def Z(U_xz, U_z): 
    return U_xz & U_z
def W(X, U_w, Z): 
    return (X & U_w )| Z
def Y(U_y, X, Z, W): 
    return (U_y & X) | (U_y & Z) | (U_y & W)


def DE_unit(U_x, U_z, U_w, U_y, U_xz):
    zee = Z(U_xz, U_z)
    # letting W be W_x0 by putting in 0 for X
    W0 = W(0, U_w, zee)
    ex = X(U_x, U_xz)
    wee = W(ex, U_w, zee)

    # plugging in 1 for x
    qty1 = Y(U_y = U_y, X = 1, Z = zee, W = W0)
    qty2 = Y(U_y = U_y, X = 0, Z = zee, W = W0)
    return (qty1 - qty2)

def IE_unit(U_x, U_z, U_w, U_y, U_xz): 
    zee = Z(U_xz, U_z)
    # letting W be W_x0 by putting in 0 for X
    W0 = W(0, U_w, zee)
    W1 = W(1, U_w, zee)
    ex = X(U_x, U_xz)
    wee = W(ex, U_w, zee)
    # plugging in 1 for x
    qty1 = Y(U_y = U_y, X = 1, Z = zee, W = W0)
    qty2 = Y(U_y = U_y, X = 1, Z = zee, W = W1)
    return (qty1 - qty2) 

def ExpSE_unit(U_x, U_z, U_w, U_y, U_xz, ex=None):
    zee = Z(U_xz, U_z)
    wee = W(ex, U_w, zee)
    return Y(U_y = U_y, X = ex, Z = zee, W = wee)

def xse_unit(U_x, U_z, U_w, U_y, U_xz):
    zee = Z(U_xz, U_z)
    W1 = W(1, U_w, zee)
    return Y(U_y = U_y, X = 1, Z = zee, W = W1)


# step 2, compute probabilities using structural basis expansion

pux = 0.4
puz = 0.8
puw = 0.2
puy = 0.3
puxz = 0.4

NDE = 0
NIE = 0
ExpSE_x0 = 0
ExpSE_x1 = 0
TV = 0
xde = 0
xie = 0
xse = 0

px1 = pux + puxz - pux * puxz  
px0 = 1 - px1

all_combo = list(itertools.product([0, 1], repeat=5))
for combo in all_combo:
    U_x, U_z, U_w, U_y, U_xz = combo
    prob_x = pux if U_x == 1 else (1 - pux)
    prob_y = puy if U_y == 1 else (1 - puy)
    prob_z = puz if U_z == 1 else (1 - puz)
    prob_w = puw if U_w == 1 else (1 - puw)
    prob_xz = puxz if U_xz == 1 else (1 - puxz)
    prob = prob_x * prob_y * prob_z * prob_w * prob_xz

    ex = X(U_x, U_xz)
    zee = Z(U_xz, U_z)
    wee = W(ex, U_w, zee)
    why = Y(U_y, ex, zee, wee)

    p_u_given_x1 = prob / px1 if ex == 1 else 0
    p_u_given_x0 = prob / px0 if ex == 0 else 0
    
    TV+= why * (p_u_given_x1 - p_u_given_x0)
    NDE += prob * DE_unit(U_x, U_z, U_w, U_y, U_xz) 
    NIE += prob * IE_unit(U_x, U_z, U_w, U_y, U_xz) 
    
    ExpSE_x0 += ExpSE_unit(U_x, U_z, U_w, U_y, U_xz, ex= 0) * (p_u_given_x0 - prob)
    ExpSE_x1 += ExpSE_unit(U_x, U_z, U_w, U_y, U_xz, ex = 1) * (p_u_given_x1 - prob)

    xde += DE_unit(U_x, U_z, U_w, U_y, U_xz) * p_u_given_x0
    xie += IE_unit(U_x, U_z, U_w, U_y, U_xz) * p_u_given_x0
    xse += xse_unit(U_x, U_z, U_w, U_y, U_xz) * (p_u_given_x0 - p_u_given_x1)
    
print('TV',TV)
print('NDE',NDE)
print('NIE',NIE)
print('ExpSE_x0',ExpSE_x0)
print('ExpSE_x1',ExpSE_x1)
print(NDE - NIE + (ExpSE_x1 - ExpSE_x0))
print(xde -xie - xse)

data = gen_data(100000)
print(np.mean(data[data['X'] == 1]['Y']) - np.mean(data[data['X'] == 0]['Y']))