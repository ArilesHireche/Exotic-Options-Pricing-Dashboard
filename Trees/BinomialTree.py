import time
from numba import njit
import numpy as np

# @njit(cache=True)
def bin_tree_amer_vectorized(S, K, T, vol, r, q, n_step, call=False):
    # Implementation of American option pricing using binomial tree

    #Computing parameters
    dt = T / n_step
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt) - d) / (u-d)
    sign = -1 if call else 1 #Considering puts as default
    
    #Computing final stock prices
    j = np.arange(n_step+1)
    log_u = np.log(u)
    log_d = np.log(d)
    log_S = np.log(S)
    log_ST = log_S + j*log_u + (n_step - j)*log_d  #Using np.log and np.exp for higher speed than exponent computations
    ST = np.exp(log_ST)
    P = np.maximum(sign*(K - ST), 0)
    
    disc = np.exp(-r*dt)
    for i in range(n_step, 0, -1): #Working backward
        j = np.arange(i)
        P = np.maximum( disc * ( (1-p) * P[:-1] + p * P[1:] ),
                       sign*(K - np.exp( log_S + j*log_u + (i-1 - j)*log_d) ) )  #On each time step, early exercise occurs if the discounted risk-neutral expectation is lower than immediate execution

    return P[0]

# @njit(cache=True)
def bin_tree_amer_numba_loop(S, K, T, vol, r, q, n_step, call=False):
    # Implementation of American option pricing using binomial tree

    #Computing parameters
    dt = T / n_step
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt) - d) / (u-d)
    sign = -1 if call else 1 #Considering puts as default
    
    #Computing final stock prices
    j = np.arange(n_step+1)

    log_u = np.log(u)
    log_d = np.log(d)
    log_S = np.log(S)

    log_ST = log_S + j*log_u + (n_step - j)*log_d  #Using np.log and np.exp for higher speed than exponent computations
    ST = np.exp(log_ST)
    P = np.maximum(sign*(K - ST), 0)
    
    disc = np.exp(-r*dt)
    for i in range(n_step, 0, -1): #Working backward
        for j in range(i):
            P[j] = np.maximum( disc * ( (1-p) * P[j] + p * P[j+1] ),
                                sign*(K - np.exp( log_S + j*log_u + (i-1 - j)*log_d) ) ) #On each time step, early exercise occurs if the discounted risk-neutral expectation is lower than immediate execution

    return P[0]

# @njit(cache=True)
def bin_tree_amer_path(S, K, T, vol, r, q, n_step, call=False):
    # Displaying the binomial tree in row format, returning arrays that will be
    # rearranged in the app.py file for Streamlit visualisation

    #Computing parameters
    dt = T / n_step
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt) - d) / (u-d)
    sign = -1 if call else 1 #Considering puts as default

    #Computing final stock prices
    j = np.arange(n_step+1)
    log_u = np.log(u)
    log_d = np.log(d)
    log_S = np.log(S)
    log_ST = log_S + j*log_u + (n_step - j)*log_d  #Using np.log and np.exp for higher speed than exponent computations
    ST = np.exp(log_ST)
    P = np.maximum(sign*(K - ST), 0)
    
    St_mat = np.full((10, 10), np.nan)
    P_mat  = np.full((10, 10), np.nan)

    disc = np.exp(-r*dt)
    for i in range(n_step, 0, -1): #Working backward
        j = np.arange(i)
        if i <= 10:
            k=10 - i
            St = np.exp( log_S + j*log_u + (i-1-j)*log_d )
            P = np.maximum( disc * ( (1-p) * P[:-1] + p * P[1:] ),
                        sign*(K - St ) )  #On each time step, early exercise occurs if the discounted risk-neutral expectation is lower than immediate execution
            St_mat[k, :i] = St
            P_mat[k, :i] = P
        else:
            P = np.maximum( disc * ( (1-p) * P[:-1] + p * P[1:] ),
                        sign*(K - np.exp( log_S + j*log_u + (i-1 - j)*log_d) ) )  #On each time step, early exercise occurs if the discounted risk-neutral expectation is lower than immediate execution
    return ST, St_mat, P_mat

print(bin_tree_amer_path(30, 25, 0.3, 0.052, 0.2, 0.3, 100, call=False)[1][-2])
#bin_tree_amer_path(S=1.61, K=1.6, T=1, vol=0.12, r=0.08, q=0.09, n_step=10, call=False)
