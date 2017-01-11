import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

raw_data = pd.read_csv("linear.csv")

# Removes rows with NaN in them

filtered_data = raw_data[~np.isnan(raw_data["y"])] 



npMatrix = np.matrix(filtered_data)
X, Y = npMatrix[:,0], npMatrix[:,1]
mdl = LinearRegression().fit(X,Y) # either this or the next line
#mdl = LinearRegression().fit(filtered_data[['x']],filtered_data.y)
m = mdl.coef_[0]
b = mdl.intercept_

# show alternate way to get equation of the line
print "formula: y = {0}x + {1}".format(m, b) # following slope intercept form 

# 
# show how to plot using non python notebooks
plt.scatter(X,Y, color='blue')
plt.plot([0,100],[b,m*100+b],'r')
plt.title('Linear Regression Example', fontsize = 20)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)