import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy

import numpy as np
import scipy
import pandas as pd

class garchThree(object):
    def __init__(self, logReturns):
        self.logReturns = logReturns * 100
        self.sigma_2 = self.garch_filter(self.garch_optimization())
        self.coefficients = self.garch_optimization()
        
    def garch_filter(self, parameters):
        "Returns the variance expression of a GARCH(3,3) process."
        # Slicing the parameters list
        omega = parameters[0]
        alpha1 = parameters[1]
        alpha2 = parameters[2]
        alpha3 = parameters[3]
        beta1 = parameters[4]
        beta2 = parameters[5]
        beta3 = parameters[6]
        
        # Length of logReturns
        length = len(self.logReturns)
        
        # Initializing an empty array
        sigma_2 = np.zeros(length)
        
        # Filling the array, if i == 0 then uses the long term variance.
        for i in range(length):
            if i == 0:
                sigma_2[i] = omega / (1 - alpha1 - alpha2 - alpha3 - beta1 - beta2 - beta3)
            else:
                sigma_2[i] = omega + alpha1 * self.logReturns[i-1]**2 + alpha2 * self.logReturns[i-2]**2 + alpha3 * self.logReturns[i-3]**2 + beta1 * sigma_2[i-1] + beta2 * sigma_2[i-2] + beta3 * sigma_2[i-3]
        
        return sigma_2 
        
    def garch_loglikehihood(self, parameters):
        "Defines the log likelihood sum to be optimized given the parameters."
        length = len(self.logReturns)
        
        sigma_2 = self.garch_filter(parameters)
        
        loglikelihood = - np.sum(-np.log(sigma_2) - self.logReturns**2 / sigma_2)
        return loglikelihood
    
    def garch_optimization(self):
        "Optimizes the log likelihood function and returns estimated coefficients"
        # Parameters initialization
        parameters = [.1, .05, .05, .05, .9, .9, .9]
        
        # Parameters optimization, scipy does not have a maximize function, so we minimize the opposite of the equation described earlier
        opt = scipy.optimize.minimize(self.garch_loglikehihood, parameters,
                                     bounds = ((.001,1),(.001,1),(.001,1),(.001,1),(.001,1),(.001,1),(.001,1)))
        
        variance = .01**2 * opt.x[0] / (1 - sum(opt.x[1:7]))   # Times .01**2 because it concerns squared returns
        
        return np.append(opt.x, variance)


# Load the stock return data
start = dt.datetime(2000,1,1)
end = dt.datetime(2022,12,31)
ftse = yf.download("^FTSE", start, end)
# calculate returns
returns = 100 * ftse.Close.pct_change().dropna()

from arch import arch_model
# Estimation using our previously coded classes
model = garchThree(returns)
# Fitting using the arch_model package
lib_model = arch_model(returns, p=3, q=3, mean = 'Zero', vol = 'GARCH')
lib_model = lib_model.fit()
# Extracting confidence intervals
conf_int = pd.DataFrame(lib_model.conf_int(alpha = .2))
# Creating the test
conf_int['garchThree'] = model.coefficients[:-1]
conf_int['Test'] = np.where(conf_int['garchThree'] < conf_int['upper'], np.where(conf_int['garchThree'] > conf_int['lower'], "Ok", "Not ok"), "Not ok") 
print()