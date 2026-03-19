# Exotic Options Pricing Dashboard
 
An interactive Streamlit dashboard for pricing exotic options using numerical methods. Built as part of a Master's in Finance at Université Paris Dauphine-PSL.
 
Most of the code is heavily commented and written to be as explicit as possible — feel free to explore the modules, and hopefully have some fun along the way. Do not hesitate to text me on LinkedIn if you find any area of improvement, I would be happy to discuss it!


Note: The app was uploaded with Python 3.11 by default to ensure a proper call of numba in my functions (@njit). It should however work with Python 3.14 on your local machine.

---
 
## Features
 
- **Black-Scholes (BSM)** — Closed-form pricing for European options with Greeks and heatmaps
- **Asian Options** — Continuous geometric (Kemna & Vorst, 1990) and arithmetic (Hull, 2000) approximations
- **Barrier Options** — Analytical pricing for knock-in and knock-out options
- **Binomial Trees** — American option pricing with full tree visualization (CRR model)
- **Monte Carlo** — Simulation-based pricing with variance reduction techniques
- **Computational efficiency** — Core numerical loops accelerated with Numba JIT compilation
 
---
 
## Tech Stack
 
| Library | Usage |
|---|---|
| `streamlit` | Dashboard and UI |
| `numpy` | Numerical computations |
| `scipy` | Statistical functions (normal CDF, PDF) |
| `plotly` | Interactive charts and heatmaps |
| `pandas` | Data display |
| `numba` | JIT compilation for performance-critical loops |
 
---
 
## Run Locally
 
**1. Clone the repository**
```bash
git clone https://github.com/ArilesHireche/Exotic-Options-Pricing-Dashboard.git
cd Exotic-Options-Pricing-Dashboard
```
 
**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
 
**3. Launch the app**
```bash
streamlit run app.py
```
 
---
 
## Project Structure
 
```
Exotic-Options-Pricing-Dashboard/
│
├── app.py                  # Main Streamlit application
├── requirements.txt
├── .gitignore
│
├── BSM/
│   └── BSM.py              # Black-Scholes, Asian & Barrier pricing functions
│
└── Trees/
    └── BinomialTree.py     # Binomial tree (CRR) with Numba acceleration
```
 
---
 
## Live Demo
 
🔗 [exotic-options-pricing.streamlit.app](https://exo-pricer-hirecheariles.streamlit.app) 
