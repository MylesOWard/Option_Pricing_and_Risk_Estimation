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
    V = np.full(shape = (N+1,M), fill_value = v0)

    # sampling correlated brownian motion
    # gives a full matrix of results for all time steps and simulations
    Z = np.random.multivariate_normal(mu, cov, (N,M))

    # first we can compute changes in stock price based using the previous volatility value
    # this is a time-addative version of the ds equation, employing Ito's Lemma 
    for i in range (1,N+1):
        # using logorithms to ensure calculations are time-addative
        S[i] = S[i-1] * np.exp((r - 0.5*V[i-1])*dt + np.sqrt(V[i-1]*dt)*Z[i-1,:,0])
        # using Euler discretisation 
        V[i] = np.maximum(V[i-1] + kappa*(theta-V[i-1])*dt + sigma*np.sqrt(V[i-1]*dt)*Z[i-1,:,1],0)

    return S,V

rho_p = 0.98
rho_n = -0.98

S_p, V_p = heston_model(S, v0, rho_p, kappa, theta, sigma, T, N, M)
S_n, V_n = heston_model(S, v0, rho_n, kappa, theta, sigma, T, N, M)

# plotting
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
time = np.linspace(0,T,N+1)

# plotting variation in stock price over time 
ax1.plot(time,S_p)
ax1.set_title("Heston Model Stock Prices")
ax1.set_xlabel("Time")
ax1.set_ylabel("Asset Prices")

# plotting the variance over time
ax2.plot(time,V_p)
ax2.set_title("Heston Model Variance Over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Variance")

# since colours aren't specified they are applied in order
# therefore outlier stock price paths will be colour-coded with outlier variance
plt.show()


    
    
