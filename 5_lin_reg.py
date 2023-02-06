"""
Script for ch5
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import graphviz as gr

png_path = "pngs/ch5/"

data = pd.read_csv("data/online_classroom.csv").query("format_blended==0")

result = smf.ols('falsexam ~ format_ol', data=data).fit()
print(result.summary().tables[1])


print(data
 .groupby("format_ol")
 ["falsexam"]
 .mean())

X = data[["format_ol"]].assign(intercep=1)
y = data["falsexam"]

def regress(y, X): 
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

beta = regress(y, X)
print(beta)

kapa = data["falsexam"].cov(data["format_ol"]) / data["format_ol"].var()
print(kapa)

e = y - X.dot(beta)
print("Orthogonality imply that the dot product is zero:", np.dot(e, X))
print(X[["format_ol"]].assign(e=e).corr())

wage = pd.read_csv("./data/wage.csv").dropna()
model_1 = smf.ols('np.log(hwage) ~ educ', data=wage.assign(hwage=wage["wage"]/wage["hours"])).fit()
print(model_1.summary().tables[1])

from matplotlib import pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

x = np.array(range(5, 20))
plt.plot(x, np.exp(model_1.params["Intercept"] + model_1.params["educ"] * x))
plt.xlabel("Years of Education")
plt.ylabel("Hourly Wage")
plt.title("Impact of Education on Hourly Wage")
plt.savefig(png_path + "hourly.png")
plt.close()
print(wage.head())


controls = ['IQ', 'exper', 'tenure', 'age', 'married', 'black',
            'south', 'urban', 'sibs', 'brthord', 'meduc', 'feduc']

X = wage[controls].assign(intercep=1)
t = wage["educ"]
y = wage["lhwage"]

beta_aux = regress(t, X)
t_tilde = t - X.dot(beta_aux)

kappa = t_tilde.cov(y) / t_tilde.var()
print(kappa)

model_2 = smf.ols('lhwage ~ educ +' + '+'.join(controls), data=wage).fit()
print(model_2.summary().tables[1])

g = gr.Digraph(format="png")
g.edge("W", "T"), g.edge("W", "Y"), g.edge("T", "Y")
g.edge("IQ", "Educ", color="red"), g.edge("IQ", "Wage", color="red"), g.edge("Educ", "Wage", color="red")
g.edge("Crime", "Police", color="red"), g.edge("Crime", "Violence", color="red"), 
g.edge("Police", "Violence", color="blue")
g.render(filename=png_path+"conf")
os.remove(png_path + "conf")


g = gr.Digraph(format="png")
g.edge("W", "Y"), g.edge("T", "Y")
g.edge("IQ", "Wage", color="red"), g.edge("Educ", "Wage", color="red")
g.render(filename=png_path+"conf2")
os.remove(png_path + "conf2")

g = gr.Digraph(format="png")
g.node("W=w"), g.edge("T", "Y")
g.node("IQ=x"), g.edge("Educ", "Wage", color="red")
g.render(filename=png_path+"conf3")
os.remove(png_path + "conf3")
