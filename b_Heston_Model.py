import matplotlib
import pandas as pd
import datetime as dt
import seaborn as sns
import numpy as np
import time
import yfinance as yf
from scipy.stats import norm
from matplotlib import pyplot as plt

# The Heston model is similar to the Black Scholes model but does not assume constant volatility
# Instead volatility is estimated using Browinian motion (the same Brownian motion used in fluid dynamics)

###

# simulation dependent parameters
# similar to Black-Scholes 

S = 100.0   # S is underlying price, the real value of the equity 

T = 1.0     # T is the time to expiration

r = 0.02    # r is the risk-free rate

N = 252     # number of time steps in simulations

M = 10000   # number of simulations

# static volatility estimate
# no strike price since we aren't pricing a specific option 

###

# Heston dependent parameters, all assuming risk-neutral conditions

kappa = 3           # mean reversion speed

theta = 0.20**2     # long term mean of variance 

v0 = 0.25**2        # initial variance

rho = 0.7           # correlation between assets and volatility 

sigma = 0.6         # volatility of volatility (metavolatility)

theta, v0

def heston_model(S, v0, rho, kappa, theta, sigma, T, N, M):

    dt = T/N
    mu = np.zeros(2)
    cov = np.array([[1, rho],
                  [rho, 1]])

    # reassign price and variances varaibles into arrays
    # these arrays store prices and variances
    S = np.full(shape = (N+1,M), fill_value = S)
    V = np.full(shape = (N+1,M), fill_value = V)

    # sampling correlated brownian motion
    Z = np.random.multivariate_normal(mu, cov (N,M))

    # first we can compute changes in volatility 

    # now we compute changes in stock price based using the previous value
    # this is a time-addative version of the ds equation, employing Ito's Lemma 
    S[i] = S[i-1] + np.exp((r-0.5*V[i-1])*dt + np.sqrt(V[i-1]*dt)*Z[i-1:,0])
    
    
