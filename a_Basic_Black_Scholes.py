import pandas as pd
import datetime as dt
import math
import numpy as np
import yfinance as yf
from scipy.stats import norm
from matplotlib import pyplot as plt

# the Black Scholes model is the most widely used formula for pricing European-style options 
# this is a deterministic version, exluding Brownian motion for the time being

# S is underlying price, the real value of the equity 
S = 52

# K is the strike price, or agreed call/put price in an option  
K = 50

# T is the time to expiration
T = 0.5

# r is the risk-free rate 
r = 0.1

# volatility (sigma)
vol = 0.2 

# volatility increases the value of both puts and calls, when volatility is high certanty has a higher premium


# d1 is the adjusted chance that the option will expire "in the money" used for hedging

d1 = (np.log(S/K) + (r + 0.5*(vol**2))*T)/(vol*math.sqrt(T))#

# risk-neutral probability that the option will be exercised (stock price > strike price)

d2 = d1 - vol*(math.sqrt(T))

# calculate call option price

C =  S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# calculate put option price

P = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

print("The value of d1 is: " + str(round(d1,4)))
print("The value of d2 is: " + str(round(d2,4)))
print("The price of this call option is $", float(round(C,2)))
print("The price of this put option is $", float(round(P,2)))
