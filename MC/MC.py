import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #Adding the parent directory to the path, to import BSM functions
from BSM.BSM import KemnaVorstGeo

def WtMC(S, K , T, vol, r, q, call=False, n_sim=10_000):
    """ Monte Carlo simulation for Asian geometric average options. 
    The payoff is max(0, sign*(G-K)) where G is the geometric average of the underlying price
    """
    z = np.random.normal(0, 1, n_sim)
    Gt = S * np.exp(T*0.5*(r-q-0.5*vol**2) + vol*np.sqrt(T/3)*z)
    sign = 1 if call else -1
    Wt = np.maximum(sign*(Gt-K), 0)*np.exp(-r*T)
    return Wt.mean(), Wt.std()/np.sqrt(n_sim), Wt, Gt

def YtMC(S, K, T, vol, r, q, call=False, n_sim=100_000, n_step=None):
    """Kemna and Vorst simulation of the discrete arithmetic average options.
    If we observe the price at close of each day, we choose dt=1/256 <=> n_step = 256*T
    """
    if n_step is None:
        n_step = int(256*T)
    else:
        n_step = int(n_step)
    dt = T/n_step

    exponent = (r-q-0.5*vol**2)*dt + vol * np.sqrt(dt) * np.random.normal(0, 1, (n_step, n_sim)) #The observation at time 0 is already observed
    dlnSt = np.cumsum( exponent , axis=0) #dln(S1) ; dln(S1)+dln(S2) ; ... ; dln(S1)+dln(S2)+...+dln(ST) along each simulation
    dlnSt = np.vstack([np.zeros((1, n_sim)), dlnSt])  #Adding ln(S0) to the series
    St = S*np.exp(dlnSt)
    At = St.sum(axis=0)/(n_step+1)

    sign = 1 if call else -1
    Yt = np.maximum(sign*(At - K), 0) * np.exp(-r*T)
    return Yt.mean(), Yt.std()/np.sqrt(n_sim)

def AsianAritVarReduc(S, K, T, vol, r, q, call=False, n_sim=1_000, n_step=None):
    """Variance reduction through control variate technique.
    Control variate method reduces variance, and compensates for discretization error 
    by using the continuous geometric formula (Horvath & Medvegyev)
    We aim for a 1 cent CI => standard error of 0.0025 that achieve rougly with a 1,000 simulations."""
    if n_step is None:
        n_step = int(256*T)
    else:
        n_step = int(n_step)
    Ganalytic = KemnaVorstGeo(S, K , T, vol, r, q, call)
    dt = T/n_step
    disc = np.exp(-r*T) #To avoid repetition

    dlnSt = (r-q-0.5*vol**2)*dt + vol * np.sqrt(dt) * np.random.normal(0, 1, (n_step, n_sim)) #The observation at time 0 is already observed
    prices = np.empty((n_step+1, n_sim)) #We create an empty array first that we'll modify instead of creating new ones later
    prices[0] = 0 #no variation of the price at time 0
    np.cumsum(dlnSt, axis=0, out=prices[1:]) #filling the other spots of the array with the cumulative log variations 
                                                #dln(S1) ; dln(S1)+dln(S2) ; ... ; dln(S1)+dln(S2)+...+dln(ST) along each simulation
    Gt = np.exp( np.log(S) + np.sum(prices, axis=0)/(n_step+1) )#We do the geometric average in log before getting back to real prices need for At
    np.exp( prices, out=prices) #We modify the same array to save memory
    prices *= S  #Getting back to real prices S*exp(dln(St))
    At = prices.sum(axis=0)/(n_step+1) #Arithmetic average

    sign = 1 if call else -1
    Yt = np.maximum(sign*(At - K), 0) * disc
    YMC = Yt.mean()

    Wt = np.maximum(sign*(Gt-K), 0) * disc
    GMC = Wt.mean()

    return YMC + (Ganalytic-GMC), (Yt - Wt).std()/np.sqrt(n_sim), At, prices
    #return f"Premium = {YMC + (Ganalytic-GMC)}, standard deviation = {(Yt - Wt).std()}, standard error = {(Yt - Wt).std()/np.sqrt(n_sim)}"

#Barrier options
def Barrier(S, K, H, T, vol, r, q, call=False, knock="Knock-In", M=100_000, n_step=None):
    """Not mandatory since we have the exact closed form solution under BSM assumptions"""
    
    if n_step is None:
        n_step = int(256*T)
    dt = T/n_step
    disc = np.exp(-r*T)
    up = S<H

    dlnSt = (r-q-0.5*vol**2)*dt + vol * np.sqrt(dt) * np.random.normal(0, 1, (n_step, M))
    prices = np.empty((n_step + 1, M))
    prices[0] = 0
    np.cumsum(dlnSt, axis=0, out=prices[1:])
    np.exp(prices, out=prices)
    prices *= S
    St = np.copy(prices) #We keep the prices for visualisation purposes

    if up:
        if knock == "Knock-In":
            prices[:, ~np.any(prices>=H, axis=0)] = 0 #If any price breaches the barrier, np.any returns True and the column will be assigned False so not 0
        else:    #Use ~ instead of not for arrays
            prices[:, np.any(prices>=H, axis=0)] = 0
    else:
        if knock == "Knock-In":
            prices[:, ~np.any(prices<=H, axis=0)] = 0
        else:
            prices[:, np.any(prices<=H, axis=0)] = 0

    sign = 1 if call else -1
    disc_payoffs = disc * np.maximum(sign*(prices[-1, :] - K), 0)
    return disc_payoffs.mean(), disc_payoffs.std()/np.sqrt(M), St




#Visualization functions
def plot_paths(St, title="Asset's price paths", n_show=None):
    """St: array of shape (n_steps+1, n_paths) containing stock prices"""
    if n_show is None:
        n_show = St.shape[1]
    n_steps = St.shape[0] - 1
    step = np.arange(n_steps + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(step, St[:, :n_show], linewidth=0.5, alpha=0.5)
    ax.axhline(St[0, 0], color='black', alpha=0.2)
    ax.set_xlabel('Step', size=12)
    ax.set_ylabel('St', size=12)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def ST_dist(ST, title="S(T) distribution"):
    mean = ST.mean()
    std = ST.std()
    se = std / np.sqrt(len(ST))
    
    log_mean = np.log(ST).mean()
    log_std = np.log(ST).std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ST, bins=40, color='blue', linewidth=0.3, edgecolor='black', density=True)

    x = np.linspace(ST.min(), ST.max(), 200)
    ax.plot(x, lognorm.pdf(x, s=log_std, scale=np.exp(log_mean)), color='black', linewidth=2, label=f'LogN(μ={log_mean:.2f}, sigma={log_std:.2f})', linestyle='--')
    
    ax.axvline(mean, color='red', linewidth=1.5, label=f'Mean = {mean:.2f}')
    ax.axvline(mean + 1.96*se, color='orange', linestyle='--', linewidth=1.5, label=f'+1.96 SE = {mean+1.96*se:.2f}')
    ax.axvline(mean - 1.96*se, color='orange', linestyle='--', linewidth=1.5, label=f'−1.96 SE = {mean-1.96*se:.2f}')
    ax.legend(fontsize=9)
    ax.set_xlabel('S(T)', size=12)
    ax.set_ylabel('Density', size=12)
    ax.set_title(title)
    plt.tight_layout()
    return fig

###Speed computation###

#Variance reduction
#t0 = time.perf_counter()
#priceVR = AsianAritVarReduc(40, 45, 1/3, 0.3, 0.05, 0, call=True, n_sim=1_000, n_step=88)[0]
#length = time.perf_counter() - t0
#print("Time with VR", length)
#print(priceVR)

#Plain MC
# t0b = time.perf_counter()
# priceMC = YtMC(40, 45, 1/3, 0.3, 0.05, 0, call=True, n_sim=100_000, n_step=88)
# lengthb = time.perf_counter() - t0b
# print("Time for classic MC", lengthb)
# print(priceMC)