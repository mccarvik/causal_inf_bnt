"""
Script for ch8
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import graphviz as gr
from linearmodels.iv import IV2SLS
import pdb
import os

pd.set_option("display.max_columns", 5)
style.use("fivethirtyeight")


png_path = "pngs/ch8/"

# ovb = ommited variable bias

g = gr.Digraph(format="png")
g.edge("ability", "educ")
g.edge("ability", "wage")
g.edge("educ", "wage")
g.render(filename=png_path+"ability")
os.remove(png_path + "ability")

# Here is where Instrumental Variables enters the picture. 
# The idea of IV is to find another variable that causes the treatment and it is only correlated with the outcome through the treatment. 
# Another way of saying this is that this instrument Z is uncorrelated with Y but is correlated with T

g = gr.Digraph(format="png")
g.edge("ability", "educ")
g.edge("ability", "wage")
g.edge("educ", "wage")
g.edge("instrument", "educ")
g.render(filename=png_path+"ability2")
os.remove(png_path + "ability2")

# To be honest, good instruments are so hard to come by that we might as well consider them miracles.
# EX: We will again try to estimate the effect of education on wage. To do so, we will use the person's quarter of birth as the instrument Z.
# This idea takes advantage of US compulsory attendance law. 
# Usually, they state that a kid must have turned 6 years by January 1 of the year they enter school. 
# For this reason, kids that are born at the beginning of the year will enter school at an older age. 
# Compulsory attendance law also requires students to be in school until they turn 16, at which point they are legally allowed to drop out.
#  The result is that people born later in the year have, on average, more years of education than those born in the beginning of the year.

g = gr.Digraph(format="png")
g.edge("ability", "educ")
g.edge("ability", "wage")
g.edge("educ", "wage")
g.edge("qob", "educ")
g.render(filename=png_path+"qob")
os.remove(png_path + "qob")

data = pd.read_csv("./data/ak91.csv")
print(data.head())

group_data = (data
              .groupby(["year_of_birth", "quarter_of_birth"])
              [["log_wage", "years_of_schooling"]]
              .mean()
              .reset_index()
              .assign(time_of_birth = lambda d: d["year_of_birth"] + (d["quarter_of_birth"])/4))

plt.figure(figsize=(15,6))
plt.plot(group_data["time_of_birth"], group_data["years_of_schooling"], zorder=-1)
for q in range(1, 5):
    x = group_data.query(f"quarter_of_birth=={q}")["time_of_birth"]
    y = group_data.query(f"quarter_of_birth=={q}")["years_of_schooling"]
    plt.scatter(x, y, marker="s", s=200, c=f"C{q}")
    plt.scatter(x, y, marker=f"${q}$", s=100, c=f"white")

plt.title("Years of Education by Quarter of Birth (first stage)")
plt.xlabel("Year of Birth")
plt.ylabel("Years of Schooling")
plt.savefig(png_path + "q_year.png")
plt.close()

factor_data = data.assign(**{f"q{int(q)}": (data["quarter_of_birth"] == q).astype(int)
                             for q in data["quarter_of_birth"].unique()})

print(factor_data.head())

first_stage = smf.ols("years_of_schooling ~ C(year_of_birth) + C(state_of_birth) + q4", data=factor_data).fit()
print("q4 parameter estimate:, ", first_stage.params["q4"])
print("q4 p-value:, ", first_stage.pvalues["q4"])

plt.figure(figsize=(15,6))
plt.plot(group_data["time_of_birth"], group_data["log_wage"], zorder=-1)
for q in range(1, 5):
    x = group_data.query(f"quarter_of_birth=={q}")["time_of_birth"]
    y = group_data.query(f"quarter_of_birth=={q}")["log_wage"]
    plt.scatter(x, y, marker="s", s=200, c=f"C{q}")
    plt.scatter(x, y, marker=f"${q}$", s=100, c=f"white")

plt.title("Average Weekly Wage by Quarter of Birth (reduced form)")
plt.xlabel("Year of Birth")
plt.ylabel("Log Weekly Earnings")
plt.savefig(png_path+"log_weekly.png")
plt.close()

reduced_form = smf.ols("log_wage ~ C(year_of_birth) + C(state_of_birth) + q4", data=factor_data).fit()
print("q4 parameter estimate:, ", reduced_form.params["q4"])
print("q4 p-value:, ", reduced_form.pvalues["q4"])
print(reduced_form.params["q4"] / first_stage.params["q4"])

# IV = instrumental variables
# Another way to get the IV estimates is by using 2 stages least squares, 2SLS. 
# With this procedure, we do the first stage like before and then run a second stage, 
# where we replace the treatment variable by the fitted values of the 1st stage

iv_by_hand = smf.ols("log_wage ~ C(year_of_birth) + C(state_of_birth) + years_of_schooling_fitted",
                     data=factor_data.assign(years_of_schooling_fitted=first_stage.fittedvalues)).fit()
print(iv_by_hand.params["years_of_schooling_fitted"])


def parse(model, exog="years_of_schooling"):
    param = model.params[exog]
    se = model.std_errors[exog]
    p_val = model.pvalues[exog]
    print(f"Parameter: {param}")
    print(f"SE: {se}")
    print(f"95 CI: {(-1.96*se,1.96*se) + param}")
    print(f"P-value: {p_val}")
    
formula = 'log_wage ~ 1 + C(year_of_birth) + C(state_of_birth) + [years_of_schooling ~ q4]'
iv2sls = IV2SLS.from_formula(formula, factor_data).fit()
parse(iv2sls)

formula = 'log_wage ~ 1 + C(year_of_birth) + C(state_of_birth) + [years_of_schooling ~ q1+q2+q3]'
iv_many_zs = IV2SLS.from_formula(formula, factor_data).fit()
parse(iv_many_zs)

formula = "log_wage ~ years_of_schooling + C(state_of_birth) + C(year_of_birth) + C(quarter_of_birth)"
ols = IV2SLS.from_formula(formula, data=data).fit()
parse(ols)

# When dealing with IV, we need to remember we are estimating the ATE indirectly. 
# Our estimates depend on both the first stage and the second stage. 
# If the impact of the treatment on the outcome is indeed strong, the second stage will also be strong. 
# However, it doesn't matter how strong the second stage is if we have a weak first stage. 
# A weak first stage means that the instrument has only a very small correlation with the treatment. 
# Therefore, we can't learn much about the treatment from the instrument.

# simulation to see standard errors
np.random.seed(12)
n = 10000
X = np.random.normal(0, 2, n) # observable variable
U = np.random.normal(0, 2, n) # unobservable (omitted) variable
T = np.random.normal(1 + 0.5*U, 5, n) # treatment
Y = np.random.normal(2 + X - 0.5*U + 2*T, 5, n) # outcome

stddevs = np.linspace(0.1, 100, 50)
Zs = {f"Z_{z}": np.random.normal(T, s, n) for z, s in enumerate(stddevs)} # instruments with decreasing \mathrm{Cov}(Z, T)
sim_data = pd.DataFrame(dict(U=U, T=T, Y=Y)).assign(**Zs)
print(sim_data.head())

# correlation decreasing:
corr = (sim_data.corr()["T"]
        [lambda d: d.index.str.startswith("Z")])
print(corr.head())

# Now, we will run one IV model per instrument we have and collect both the ATE estimate and the standard error.
se = []
ate = []
for z in range(len(Zs)):
    formula = f'Y ~ 1 + X + [T ~ Z_{z}]'
    iv = IV2SLS.from_formula(formula, sim_data).fit()
    se.append(iv.std_errors["T"])
    ate.append(iv.params["T"])

plot_data = pd.DataFrame(dict(se=se, ate=ate, corr=corr)).sort_values(by="corr")
plt.scatter(plot_data["corr"], plot_data["se"])
plt.xlabel("Corr(Z, T)")
plt.ylabel("IV Standard Error");
plt.title("Variance of the IV Estimates by 1st Stage Strength")
plt.savefig(png_path + "var_iv_ests.png")
plt.close()

plt.scatter(plot_data["corr"], plot_data["ate"])
plt.fill_between(plot_data["corr"],
                 plot_data["ate"]+1.96*plot_data["se"],
                 plot_data["ate"]-1.96*plot_data["se"], alpha=.5)
plt.xlabel("Corr(Z, T)")
plt.ylabel("$\hat{ATE}$");
plt.title("IV ATE Estimates by 1st Stage Strength");
plt.savefig(png_path + "ate_ests.png")
plt.close()

# As we can see in the plots above, estimates vary wildly when the correlation between T and Z is weak. 
# This is because the SE also increases a lot when the correlation is low
# Another thing to notice is that 2SLS is biased! 
# Even with high correlation, the parameter estimate still does not reach the true ATE of 2.0. 
# Actually, 2.0 is not even in the 95% CI!
