from statistics import correlation
import numpy as np
import scipy.stats as sps
import plotly.graph_objects as go

def european(S, K , T, vol, r, q, call=False) :
    d1 = ( np.log(S/K) + (r-q+0.5*vol**2)*T ) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    sign = 1 if call else -1
    return sign * ( S*np.exp(-q*T) * sps.norm.cdf(sign*d1) - K*np.exp(-r*T)*sps.norm.cdf(sign*d2) )

def european_vect(S, K , T, vol, r, q, call=False):
    S, K , T, vol, r, q = map(np.asarray, (S, K, T, vol, r, q))  #Allow to handle array parameters
    
    d1 = ( np.log(S/K) + (r-q+0.5*vol**2)*T ) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    sign = 1 if call else -1
    return sign * ( S*np.exp(-q*T) * sps.norm.cdf(sign*d1) - K*np.exp(-r*T)*sps.norm.cdf(sign*d2) )
    
def BlackModel(F, K , T, vol, r, call=False):
    d1 = ( np.log(F/K) + T*vol**2/2 ) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    sign = 1 if call else -1
    return sign * np.exp(-r*T) * ( F*sps.norm.cdf(sign*d1) - K*sps.norm.cdf(sign*d2) )

def AsianContGeoBouz(S, K , T, vol, r, q, call=False):
    growth = (r - q - vol**2/6)/2
    F = S*np.exp(growth*T)
    sigma = vol / np.sqrt(3)
    d1 = ( np.log(F/K) + T*sigma**2/2 ) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    sign = 1 if call else -1
    return sign * np.exp(-r*T) * ( F * sps.norm.cdf(sign*d1) - K*sps.norm.cdf(sign*d2))

def KemnaVorstGeo(S, K , T, vol, r, q, call=False):
    """ Kemna and Vorst adjustment for geometric average otpions
    see Kemna, A. G. Z., & Vorst, A. C. F. (1990). A pricing method for options based on average asset values. Journal of Banking & Finance, 14(1), 113-129.
    """

    dstar = (r - vol**2/6)*T/2
    d = ( np.log(S/K) + (T*(r + vol**2/6))/2 ) / vol / np.sqrt(T/3)
    sign = 1 if call else -1
    return sign * np.exp(-r*T) * ( np.exp(dstar) * S *sps.norm.cdf(sign*d) - K*sps.norm.cdf(sign*(d-vol*np.sqrt(T/3)) ) )

def AsianContAritAdjHull(S, K , T, vol, r, q, call=False):
    """ Fitting a lognomral distribution with the first two moments
    of the arithmetic average, to then use Black's formula. """

    F = (np.exp(T*(r-q)) - 1) * S / (T*(r-q))
    #First moment for a continuous arithmetic avg
    M2 = ( 2*np.exp(T*(2*(r-q)+vol**2))*S**2 ) / ( (r-q+vol**2)*(2*r-2*q+vol**2)*T**2 ) + 2*S**2/(r-q)/T**2 *(1/(2*(r-q)+vol**2) - np.exp(T*(r-q))/(r-q+vol**2) )
    #Second moment for a continuous arithmetic avg
    sigma = np.sqrt( np.log(M2/F**2) / T )
    d1 = ( np.log(F/K) + T*sigma**2/2 ) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    sign = 1 if call else -1
    return sign * np.exp(-r*T) * ( F*sps.norm.cdf(sign*d1) - K*sps.norm.cdf(sign*d2) )

def BSM_heatmap(price_grid, Ks, Ts, title="Premium (with constant vol)"):
    fig = go.Figure(go.Heatmap(z=price_grid, x = Ks, y=Ts, colorscale="RdYlGn",  #Building the heatmap
                            text=np.round(price_grid, 4), texttemplate="%{text:.4f}",  #Displaying the values on the heatmap
                            hovertemplate="K=%{x:.4f}<br>T=%{y:.4f}<br>Premium=%{z:.4f}<extra></extra>"))
                            #Defines the display format when hovering over points on the graph.

    fig.update_layout(xaxis_title="Strikes (K)", yaxis_title="Maturity (T)", 
                    margin=dict(l=20, r=20, t=30, b=20), title = title)

    #Highlighting the chosen otpion price
    dx, dy = Ks[1] - Ks[0], Ts[1] - Ts[0]
    fig.add_shape(type="rect",x0=Ks[4]-dx/2, x1=Ks[4]+dx/2, y0=Ts[4]-dy/2, y1=Ts[4]+dy/2,
                line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")

    Zmark = np.full_like(price_grid, np.nan, dtype=float) #Return a full array with the same shape and type as a given array.
    Zmark[4, 4] = price_grid[4, 4]
    fig.add_trace(go.Heatmap(z=Zmark, x=Ks, y=Ts, showscale=False, hoverinfo="skip",
                            colorscale=[[0, "rgba(255,255,255,0.0)"], [1, "rgba(255,255,255,0.35)"]]
                            ))
    return fig

#Barrier options
def barrier_price(S, K, H, T, vol, r, q, call=False, knock="Knock-In"):
    """
    We use the formulas from Haug's book "The complete guide to option pricing formulas" (2007),
    building the formulas in all of the cases from the same A, B, C, D blocks aiming for optimal efficiency (fewest CDF calls)
    """

    N = sps.norm.cdf
    up  = H > S
    sign = 1 if call else -1
    is_down = -1 if up else 1
    is_in = knock == "Knock-In"
    is_KgH =  bool(np.all(H <= K)) #accepts arrays too, for later price grid visualisation

    #Trivial cases
    if call and up and is_KgH:
        return european(S, K , T, vol, r, q, call) if is_in else 0.0
    
    if not call and not up and not is_KgH:
        return european(S, K , T, vol, r, q, call) if is_in else 0.0
    
    #Variables for complex formulas
    b = r-q
    lamb = (b + 0.5 * vol**2)/ vol**2
    sigT  = vol * np.sqrt(T) #Creating this variable to avoid redundance

    y = np.log(H**2/(S*K))/sigT + lamb * sigT
    x1 = np.log(S/H)/sigT + lamb * sigT
    y1 = np.log(H/S)/sigT + lamb * sigT

    HSl = (H/S)**(2*lamb)
    HS2l = (H/S)**(2*lamb-2)
    discS = S * np.exp(-q*T)
    discK = K * np.exp(-r*T)

    #Segmentation of the different cases
    def B():
        return sign * (discS * N(sign * x1) - discK * N(sign * (x1 - sigT)))
 
    def C():
        return sign * (discS * HSl * N(is_down * y) - discK * HS2l * N(is_down * (y - sigT)))
 
    def D():
        return sign * (discS * HSl * N(is_down * y1) - discK * HS2l * N(is_down * (y1 - sigT)))
 
    # Optimal formula selection (fewest CDF calls)
    # For each case, we compute whichever of KI/KO costs fewer CDF evaluations and derive the other via in-out parity.
 
    if call:
        if not up and is_KgH:    # down call, K >= H
            if is_in:
                return C()
            else:
                return european(S, K, T, vol, r, q, call) - C()
 
        else:    # down call K<H  OR  up call K<H
            if not is_in:
                return B()-D()   # KO=B-D (4 cdf)
            else:
                return european(S, K, T, vol, r, q, call) -( B()-D() ) # 6 cdf
 
    else:
        if (not up and is_KgH) or (up and not is_KgH):  # down put K>=H  OR  up put K<H
            if is_in:
                return B() - C() + D()    # KI=B-C+D (6 cdf)
            else:
                return european(S, K, T, vol, r, q, call) - (B() - C() + D())
            
 
        else:    # up put K>=H
            if not is_in:
                return B()-D()   # KO=B-D (4 cdf)
            else :
                return european(S, K, T, vol, r, q, call) - (B()-D())

def barrier_grid(S, Ks, H, Ts, vol, r, q, call, knock):
    """Adjustment for arrays inputs in perspective of heatmap visualisation."""
    result = np.empty_like(Ks)
    mask = Ks >= H #We check the barrier vs K condition
    for m in [mask, ~mask]:  #We consider when first S>K and when K<S
        if m.any(): #Not losing time if all K verify the same condition vs H
            result[m] = barrier_price(S, Ks[m], H, Ts[m],vol, r, q, call=call, knock=knock)
    return result  #returns a grid with the different prices, ready for heatmap visualisation

#WO/BO options on two assets. See MC.py file for three assets valuation.
def WOC(S1, S2, K, T, vol1, vol2, r, rho):

    sig  = np.sqrt(vol1**2 + vol2**2 - 2 * rho * vol1 * vol2)
    sT   = sig * np.sqrt(T)
    s1T  = vol1 * np.sqrt(T)
    s2T  = vol2 * np.sqrt(T)

    rho1 = (rho * vol2 - vol1) / sig
    rho2 = (rho * vol1 - vol2) / sig

    M2 = sps.multivariate_normal.cdf

    return ( S1 * M2( [s1T + (np.log(S1 / K)  + (r - 0.5 * vol1**2) * T) / s1T , (np.log(S2/S1)-np.sqrt(T)*0.5*sig**2)/sT] , mean=[0,0], cov=[[1,rho1],[rho1,1]]) 
    + S2 * M2( [s2T + (np.log(S2 / K)  + (r - 0.5 * vol2**2) * T) / s2T , (np.log(S1/S2)-np.sqrt(T)*0.5*sig**2)/sT] , mean=[0,0], cov=[[1,rho2],[rho2,1]])
    - K  * np.exp(-r * T)  * M2([(np.log(S1 / K)  + (r - 0.5 * vol1**2) * T) / s1T , (np.log(S2 / K)  + (r - 0.5 * vol2**2) * T) / s2T], mean=[0,0], cov=[[1,rho],[rho,1]])
    )

def pv_min(S1, S2, T, vol1, vol2, rho):
    """The premium of a WOcall striking at 0, is the present value of the asset that will be the lowest at maturity.
    Thus, we must weight the current price of each asset by the probability that they'll be the lowest at maturity, keeping in mind
    We assume no dividend in these functions WO/BO Black Scholes Valuation functions. It'll be introduced in MC simulations.
    The effect of r cancels out since it is the same on the two assets (identically, the discount term K*exp(-rT) vanishes in WOC formula when K=0)
    """

    sig = np.sqrt(vol1**2 + vol2**2 - 2 * rho * vol1 * vol2)
    sT  = sig * np.sqrt(T)
 
    #In BSM formula, we know that N(d2) is the probability that S>K at T and S*N(d1) the expected value of the spot conditional to S>K
    #We can replace K by S2 in the formula such that N(d2) will give us the probability that S1>S2 at maturity, but considering S2 stochasticity unlike K
    d = (np.log(S1/S2)+0.5*T*sig**2)/sT #d2 for the process of the difference of correlated process which is itself a process whose standard deviation is sig
 
    N = sps.norm.cdf
    # pv_max = S1 * N(d) + S2 * N(-d + sT) #We add sT because of the change of numeraire between S1 measure and S2 measure
    pv_min = S1 * N(-d) + S2 * N(d - sT) #The BOcall/BOput parity requires BOcall(0)=pv_max, but this adds two cdf while we could use the WO/BO parity
 
    return pv_min #, pv_max  -> we only require pv_min to find WOput using WOcall/WOput parity, then for BOput we'll use the parity with WOput, sving two cdf computations

# print( WOC(30, 35, 20, 0.5, 0.2, 0.15, 0.05, 0.2),
#        european(30, 20, 0.5, 0.2, 0.05, 0, call=True),
#        european(30, 20, 0.5, 0.2, 0.05, 0, call=True) + european(35, 20, 0.5, 0.15, 0.05, 0, call=True) - WOC(30, 35, 20, 0.5, 0.2, 0.15, 0.05, 0.2) ) #WO/BO parity

# print(pv_min(30, 35, 0.5, 0.2, 0.15, 0.2), WOC(30, 35, 0.0001, 0.5, 0.2, 0.15, 0.05, 0.2), WOC(30, 35, 0.0001, 0.5, 0.2, 0.15, 0.15, 0.2))

def WO_BO_options(S1, S2, K, T, vol1, vol2, r, rho, call=False, best_of=False):

    sig  = np.sqrt(vol1**2 + vol2**2 - 2 * rho * vol1 * vol2)
    sT   = sig * np.sqrt(T)
    s1T  = vol1 * np.sqrt(T)
    s2T  = vol2 * np.sqrt(T)

    rho1 = (rho * vol2 - vol1) / sig
    rho2 = (rho * vol1 - vol2) / sig

    M2 = sps.multivariate_normal.cdf

    WOC = ( S1 * M2( [s1T + (np.log(S1 / K)  + (r - 0.5 * vol1**2) * T) / s1T , (np.log(S2/S1)-np.sqrt(T)*0.5*sig**2)/sT] , mean=[0,0], cov=[[1,rho1],[rho1,1]]) 
    + S2 * M2( [s2T + (np.log(S2 / K)  + (r - 0.5 * vol2**2) * T) / s2T , (np.log(S1/S2)-np.sqrt(T)*0.5*sig**2)/sT] , mean=[0,0], cov=[[1,rho2],[rho2,1]])
    - K  * np.exp(-r * T)  * M2([(np.log(S1 / K)  + (r - 0.5 * vol1**2) * T) / s1T , (np.log(S2 / K)  + (r - 0.5 * vol2**2) * T) / s2T], mean=[0,0], cov=[[1,rho],[rho,1]])
    )

    if not(best_of) and call:
        return WOC
    elif not(best_of) and not call:  #Using C/P parity on worst off options, with pv_min = WOC(K=0)
        return WOC + K * np.exp(-r * T) - pv_min(S1, S2, T, vol1, vol2, rho)  #WOput = K*exp(-rT) - WOcall(K=0) + WOcall(K)
    elif best_of and call:
        return european(S1, K, T, vol1, r, 0, True) + european(S2, K, T, vol2, r, 0, True) - WOC #BOC = call on each asset - Call on the worst off
    else:  #We derive BOput from BOput = Vanilla(S1) + Vanilla(S2) - WOput ; reminding that WOput = K*exp(-rT) - WOcall(K=0) + WOcall(K)
        return european(S1, K, T, vol1, r, 0, False) + european(S2, K, T, vol2, r, 0, False) - (WOC + K * np.exp(-r * T) - pv_min(S1, S2, T, vol1, vol2, rho))




# def WorstOff(S1, S2, K, T, vol1, vol2, r, q1, q2, rho, call=False):
#     N2 = sps.multivariate_normal.cdf
#     sigma = np.sqrt(vol1**2 + vol2**2 - 2*rho*vol1*vol2)
#     s1T = vol1*np.sqrt(T)
#     s2T = vol2*np.sqrt(T)
#     gam1 = 
#     gam2 = 
#     WOC = S1 * np.exp(-q1 * T) * N2([d1, ], mean=[0,0], cov=[[1,-rho1],[-rho1,1]]) 
#     + S2 * np.exp(-q2 * T) * N2([d2, d-sT], mean=[0,0], cov=[[1,-rho2],[-rho2,1]])
#     - K  * np.exp(-r * T)  * N2([d1-s1T, d2-s2T], mean=[0,0], cov=[[1,rho],[rho,1]])

