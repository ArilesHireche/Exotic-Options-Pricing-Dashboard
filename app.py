import streamlit as st
from Trees.BinomialTree import bin_tree_amer_path
from Trees.BinomialTree import bin_tree_amer_numba_loop
from BSM.BSM import european_vect, KemnaVorstGeo, barrier_price, barrier_grid, WO_BO_options
from BSM.BSM import BSM_heatmap
from MC.MC import AsianAritVarReduc
from MC.MC import plot_paths, ST_dist
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go   #check if truly mandatory as imported in BSM.py

#Setting titles
st.set_page_config(page_title="Exotic Option Pricing Dashboard", page_icon="💲", layout ="wide")
st.title("Exotic Option Pricing Dashboard")
#st.tabs(["outputs", "surface", "hedging"])
option_type = st.selectbox("Select Option Type", options=["American", "Barrier", "Asian", "European", "BO/WO"], key="opt_type")

#Storing the intent of buttons choices to get faster
#if "show_tree_viz" not in st.session_state:
    #st.session_state.show_tree_viz = False
if "show_time_viz" not in st.session_state:
    st.session_state.show_time_viz = False

#Option-specific parameters
def barrier_params():
    st.sidebar.subheader("Barrier Option Parameters")
    return {"knock": st.sidebar.segmented_control("Knock", options=["Knock-In", "Knock-Out(default)"], key="knock"),
            "H": st.sidebar.number_input("Barrier Level", min_value=0.01, key="H")
    }

#barrier_type = st.sidebar.selectbox("Select Barrier Type", options=["Up-and-In", "Up-and-Out", "Down-and-In", "Down-and-Out"], key="barrier_type")
#barrier_level = st.sidebar.number_input("Barrier Level", min_value=0.0, key="barrier_level", value=110.0)
#return barrier_type, barrier_level

def asian_params():
    st.sidebar.subheader("Asian Option Parameters")
    return { "avg_type": st.sidebar.segmented_control("Average Type", options=["Arithmetic (default)", "Geometric"], key="avg_type"),  #"n_obs": st.sidebar.number_input("Number of observations", min_value=1, step=1 key="n_obs") ,"avg_start": st.sidebar.number_input("Averaging starts (years from now)", min_value=0.0, value=0.0, key="avg_start"),
    }

def bowo_params():
    st.sidebar.subheader("BO/WO Option Parameters")
    return { "bo/wo": st.sidebar.segmented_control("Target preformer", options=["Best-of", "Worst-of(default)"], key="bo/wo"),
            #"Number of assets": st.sidebar.segmented_control("2", "3", key ="n_assets"),
            "S2": st.sidebar.number_input("Spot2 (S2)", min_value=0.0001, value=100., key="S2", step=0.0001, format="%.4f"),
            "vol2": st.sidebar.number_input("Volatility2 (σ2)", min_value=0., value=0.2, key="vol2", step=0.0001, format="%.4f"),
            "rho": st.sidebar.number_input("Correlation (rho)", min_value=-0.9999, max_value=0.9999, value=0., key="rho", step=0.0001, format="%.4f")
    }

# def 3assets_params():
#     return { "S2": st.sidebar.number_input("Spot2 (S2)", min_value=0.0, key="S2", step=0.0001, format="%.4f"),
#              "vol2": st.sidebar.number_input("Volatility2 (σ2)", min_value=0., value=0.2, key="vol2", step=0.0001, format="%.4f"),
#              "q2": st.sidebar.number_input("Yield2 (q2)", min_value=0.0, key="q2", step=0.0001 format="%.4f"),
#              "S3": st.sidebar.number_input("Spot3 (S3)", min_value=0.0, key="S3", step=0.0001, format="%.4f"),
#              "vol3": st.sidebar.number_input("Volatility3 (σ3)", min_value=0., value=0.2, key="vol3", step=0.0001, format="%.4f"),
#              "q3": st.sidebar.number_input("Yield3 (q3)", min_value=0.0, key="q3", step=0.0001 format="%.4f"),
#              "rho12": st.sidebar.number_input("Correlation12 (rho12)", min_value=-1.0, max_value=1.0, key="rho12", step=0.0001, format="%.4f")
#              "rho13": st.sidebar.number_input("Correlation13 (rho13)", min_value=-1.0, max_value=1.0, key="rho13", step=0.0001, format="%.4f")
#              "rho23": st.sidebar.number_input("Correlation23 (rho23)", min_value=-1.0, max_value=1.0, key="rho23", step=0.0001, format="%.4f")
# }

#Common parameters
def common_params():
    return {"S" : st.sidebar.number_input("Spot (S)", min_value=0.0001, key ="S", value=100.0, step=0.0001, format="%.4f"),
            "K" : st.sidebar.number_input("Strike(K)", min_value=0.0001, key="K", value=100.0, step=0.0001, format="%.4f"),
            "T" : st.sidebar.number_input("Maturity (in years)", min_value=0.0, value=1.0, key="T", step=0.0001, format="%.4f"),
            "vol" : st.sidebar.number_input("Volatility (σ)", min_value=0., value=0.20, key="vol", step=0.0001, format="%.4f"),
            "r" : st.sidebar.number_input("Rate (r)", value=0.02, key="r", step=0.0001, format="%.4f"),
            "q" : st.sidebar.number_input("Yield (q)", min_value=0.0, key="q", step=0.0001, format="%.4f")
            }

#Building the parameters dictionary
param_builder = {
    "Asian": asian_params,
    "Barrier": barrier_params,
    "BO/WO" : bowo_params
}

params = {"cp": st.sidebar.segmented_control("Call / Put", options=["Call", "Put (default)"], key="cp")}
        #, "Speed": st.sidebar.segmented_control("Pricing Speed", options=["Fast", "Accurate"], key="speed"),}
params |= common_params()
params |= param_builder.get(option_type, lambda: {})()


c1, c2 = st.columns(2) #Displaying side by side premium and computation time
#Binomial Tree pricing
if option_type == "American":
    t0 = time.perf_counter()
    price = bin_tree_amer_numba_loop(S=params["S"], K=params["K"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], n_step=100, call=(params["cp"]=="Call"))
    length = time.perf_counter() - t0
    with c1:
        st.metric("Premium", f"{price:.4f}")
        st.caption(f"Method: Binomial Tree")

    #Computation time button
    with c2:
        if st.button("Run time"):
            st.session_state.show_time_viz = True
        if st.session_state.show_time_viz:
            st.info(f"Computation time: {length:.6f} seconds")

    #Visualisation button
    show_tree_viz = st.toggle("Visualisation")
    if show_tree_viz:
        st.subheader("Binomial Tree Visualisation")
    
        S_T, St_vals, P_vals = bin_tree_amer_path(S=params["S"], K=params["K"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], n_step=100, call=(params["cp"]=="Call"))
        
        df_St = pd.DataFrame(St_vals, index=range(10, 0, -1)).replace({np.nan: ""})
        df_P  = pd.DataFrame(P_vals, index=range(10, 0, -1)).replace({np.nan: ""})

        st.caption("Note: For visualisation purposes, the number of steps is set to 10.")

        st.subheader("Underlying (last 10 steps)")
        st.dataframe(df_St, use_container_width=True)

        st.subheader("Option value (last 10 steps)")
        st.dataframe(df_P, use_container_width=True)
        #if params["n_step"]<=30 : Use later when offering n_stesp choice to users
            #st.write("Expected spot at maturity ( T =", np.round(params["T"], 4), ") under the risk neutral measure.\n", ST, "\n")

elif option_type == "Asian":
    if params["avg_type"] == "Geometric": 
        t0 = time.perf_counter()
        price = KemnaVorstGeo(S = params["S"], K=params["K"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], call=(params["cp"]=="Call"))
        length = time.perf_counter() - t0
        method =" Kemna and Vorst adjustment (1990)"

        with c1:
            st.metric("Premium", f"{price:.4f}")
            st.caption(f"Method : {method} ")

    else:    
        t0 = time.perf_counter()
        price, se = AsianAritVarReduc(S = params["S"], K=params["K"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], call=(params["cp"]=="Call"))[:2]
        length = time.perf_counter() - t0
        method = "Variance reduction with control variate technique (Horvath & Medvegyev, 2002)"

        with c1:   
            st.metric("Premium", f"{price:.4f}", help=f"SE: {se:.6f} | 95% CI: ±{1.96*se:.4f}")
            st.caption(f"Method : {method} ")

    #Computation time button
    with c2:
        if st.button("Run time"):
            st.session_state.show_time_viz = True
        if st.session_state.show_time_viz:
            st.info(f"Computation time: {length:.6f} seconds")

    #Visualisation button
    show_asian_viz = st.toggle("Visualisation")
    if show_asian_viz:

        if params["avg_type"] == "Geometric":
            st.subheader("BSM Option Premium Visualisation")

            #Building the K-T grid
            K, T =  params["K"], params["T"]
            Ks, Ts = np.linspace(1.2*K, 0.8*K, 9), np.linspace(1.2*T, 0.8*T, 9)
            K_grid, T_grid = np.meshgrid(Ks, Ts, indexing='xy') #Return a list of coordinate matrices from coordinate vectors.

            #Call visualisation
            price_grid_call = KemnaVorstGeo(S = params["S"], K=K_grid, T=T_grid, vol=params["vol"], r=params["r"], q=params["q"], call=True) #Building the prices grid

            fig = BSM_heatmap(price_grid_call, Ks, Ts, title=f"{params["avg_type"]} average Call Premium (with constant vol)")
            
            tab1, tab2 = st.columns(2) #Displaying side by side the visualisation
            with tab1:
                st.plotly_chart(fig, use_container_width=True)

            #Put visualisation
            price_grid_put = KemnaVorstGeo(S = params["S"], K=K_grid, T=T_grid, vol=params["vol"], r=params["r"], q=params["q"], call=False)
            fig = BSM_heatmap(price_grid_put, Ks, Ts, title=f"Geometric average Put Premium (with constant vol)")
            with tab2:
                st.plotly_chart(fig, use_container_width=True)

        else:
            #MC Path
            st.subheader("Monte Carlo Simulation of Asset Price Paths")
            tab1, tab2 = st.columns(2)
            with tab1:
                st.pyplot(plot_paths(AsianAritVarReduc(S = params["S"], K=params["K"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], call=(params["cp"]=="Call"))[3]))
            with tab2:
                st.pyplot(ST_dist(AsianAritVarReduc(S = params["S"], K=params["K"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], call=(params["cp"]=="Call"))[2], title="Arithmetic average distribution"))

elif option_type == "Barrier":
    t0 = time.perf_counter()
    price = barrier_price(S = params["S"], K=params["K"], H=params["H"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], call=(params["cp"]=="Call"), knock=params["knock"])
    length = time.perf_counter() - t0
    with c1:    
        st.metric("Premium", f"{price:.4f}")
        st.caption(f"Method : Black-Scholes-Merton Formula")

    #Computation time button
    with c2:
        if st.button("Run time"):
            st.session_state.show_time_viz = True
        if st.session_state.show_time_viz:
            st.info(f"Computation time: {length:.6f} seconds")

    #Visualisation button
    #Visualisation button
    show_heatmap_viz = st.toggle("Visualisation")
    if show_heatmap_viz:
        st.subheader("BSM Option Premium Visualisation")

        #Building the K-T grid
        K, T =  params["K"], params["T"]
        Ks, Ts = np.linspace(1.2*K, 0.8*K, 9), np.linspace(1.2*T, 0.8*T, 9)
        K_grid, T_grid = np.meshgrid(Ks, Ts, indexing='xy') #Return a list of coordinate matrices from coordinate vectors.

        #Call visualisation
        price_grid_call = barrier_grid(S = params["S"], Ks=K_grid, H=params["H"], Ts=T_grid, vol=params["vol"], r=params["r"], q=params["q"], call=True, knock = params["knock"]) #Building the prices grid
        fig = BSM_heatmap(price_grid_call, Ks, Ts, title=f"{params["knock"] if params["knock"] == "Knock-In" else "Knock-Out"} Call Premium (with constant vol)")
        
        tab1, tab2 = st.columns(2) #Displaying side by side the visualisation
        with tab1:
            st.plotly_chart(fig, use_container_width=True)

        #Put visualisation
        price_grid_put = barrier_grid(S = params["S"], Ks=K_grid, H=params["H"], Ts=T_grid, vol=params["vol"], r=params["r"], q=params["q"], call=False, knock = params["knock"])
        fig = BSM_heatmap(price_grid_put, Ks, Ts, title=f"{params["knock"] if params["knock"] == "Knock-In" else "Knock-Out"} Put Premium (with constant vol)")
        with tab2:
           st.plotly_chart(fig, use_container_width=True)

elif option_type == "BO/WO":
    t0 = time.perf_counter()
    price = WO_BO_options(S1 = params["S"], S2=params["S2"], K=params["K"], T=params["T"], vol1=params["vol"], vol2=params["vol2"], r=params["r"], rho=params["rho"], call=(params["cp"]=="Call"), best_of=(params["bo/wo"]=="Best-of"))
    length = time.perf_counter() - t0
    with c1:    
        st.metric("Premium", f"{price:.4f}", help="")
        st.caption(f"Method : Black-Scholes-Merton Formula")

    #Computation time button
    with c2:
        if st.button("Run time"):
            st.session_state.show_time_viz = True
        if st.session_state.show_time_viz:
            st.info(f"Computation time: {length:.6f} seconds")

    #Visualisation button
    # show_heatmap_viz = st.toggle("Visualisation")
    # if show_heatmap_viz:

elif option_type == "European":
    t0 = time.perf_counter()
    price = european_vect(S = params["S"], K=params["K"], T=params["T"], vol=params["vol"], r=params["r"], q=params["q"], call=(params["cp"]=="Call"))
    length = time.perf_counter() - t0
    with c1:    
        st.metric("Premium", f"{price:.4f}")
        st.caption(f"Method : Black-Scholes-Merton Formula")

    #Computation time button
    with c2:
        if st.button("Run time"):
            st.session_state.show_time_viz = True
        if st.session_state.show_time_viz:
            st.info(f"Computation time: {length:.6f} seconds")

    #Visualisation button
    show_heatmap_viz = st.toggle("Visualisation")
    if show_heatmap_viz:
        st.subheader("BSM Option Premium Visualisation")

        #Building the K-T grid
        K, T =  params["K"], params["T"]
        Ks, Ts = np.linspace(1.2*K, 0.8*K, 9), np.linspace(1.2*T, 0.8*T, 9)
        K_grid, T_grid = np.meshgrid(Ks, Ts, indexing='xy') #Return a list of coordinate matrices from coordinate vectors.

        #Call visualisation
        price_grid_call =european_vect(S = params["S"], K=K_grid, T=T_grid, vol=params["vol"], r=params["r"], q=params["q"], call=True) #Building the prices grid
        fig = BSM_heatmap(price_grid_call, Ks, Ts, title="Call Premium (with constant vol)")
        
        tab1, tab2 = st.columns(2) #Displaying side by side the visualisation
        with tab1:
            st.plotly_chart(fig, use_container_width=True)

        #Put visualisation
        price_grid_put = european_vect(S = params["S"], K=K_grid, T=T_grid, vol=params["vol"], r=params["r"], q=params["q"], call=False)
        fig = BSM_heatmap(price_grid_put, Ks, Ts, title="Put Premium (with constant vol)")
        with tab2:
            st.plotly_chart(fig, use_container_width=True)

#with st.sidebar:

# def tree_params():   this is a pricing method, so eventually select it once users will be able to choose the preffered pricing method.
#    st.sidebar.subheader("Tree parameters")
#    return st.sidebar.number_input("Number of steps", min_value=1, value=100, key="n_steps")