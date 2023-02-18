"""
Script for ch14
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import graphviz as gr
from linearmodels.datasets import wage_panel

pd.set_option("display.max_columns", 6)
style.use("fivethirtyeight")

png_path = "pngs/ch14/"


# What we can't do is assign the treatment to units based on how the outcome is growing

# Methods like propensity score, linear regression and matching are very good at controlling for confounding in non-random data, 
# but they rely on a key assumption: conditional unconfoundedness
# To put it in words, they require that all the confounders are known and measured, 
# so that we can condition on them and make the treatment as good as random
# One major issue with this is that sometimes we simply can't measure a confounder. 
# Ex: figuring out the impact of marriage on men's earnings
# It's a well known fact in economics that married men earn more than single men. 
# However, it is not clear if this relationship is causal or not. 
# It could be that more educated men are both more likely to marry and more likely to have a high earnings job, 
# which would mean that education is a confounder of the effect of marriage on earnings.

# But another confounder could be beauty. 
# It could be that more handsome men are both more likely to get married and more likely to have a high paying job. 
# Unfortunately, beauty is one of those characteristics like intelligence. It's something we can't measure very well.
# This puts us in a difficult situation, because if we have unmeasured confounders, we have bias. 

# The trick is to see that, by zooming in a unit and tracking how it evolves over time, 
# we are already controlling for anything that is fixed over time. 
# That includes any time fixed unmeasured confounders. 
# We can already know that the increase in income over time cannot be due to an increase in beauty, 
# simply because beauty stays the same (it is time fixed after all). 
# The bottom line is that even though we cannot control for beauty, since we can't measure it, 
# we can still use the panel structure so it is not a problem anymore

# Think about it. We can't measure attributes like beauty and intelligence, 
# but we know that the person who has them is the same individual across time
# All we need to do is create dummy variables indicating that person and add that to a linear model
# Adding this unit dummy is what we call a fixed effect model

data = wage_panel.load()
print(data.head())

mod = smf.ols("lwage ~ C(year)", data=data).fit()
print(mod.summary().tables[1])
print(data.groupby("year")["lwage"].mean())

# wipes out all unobserved that are constant across time. 
# Not just do the unobserved variables vanish. 
# This happens to all the variables that are constant in time. 
# For this reason, you can't include any variables that are constant across time, 
# as they would be a linear combination of the dummy variables and the model wouldn't run.
print(data.groupby("nr").std().sum())


Y = "lwage"
T = "married"
X = [T, "expersq", "union", "hours"]
mean_data = data.groupby("nr")[X+[Y]].mean()
print(mean_data.head())

demeaned_data = (data
               .set_index("nr") # set the index as the person indicator
               [X+[Y]]
               - mean_data) # subtract the mean data
print(demeaned_data.head())

# Finally, we can run our fixed effect model on the time-demeaned data.
mod = smf.ols(f"{Y} ~ {'+'.join(X)}", data=demeaned_data).fit()
print(mod.summary().tables[1])

from linearmodels.panel import PanelOLS
mod = PanelOLS.from_formula("lwage ~ expersq+union+married+hours+EntityEffects",
                            data=data.set_index(["nr", "year"]))
result = mod.fit(cov_type='clustered', cluster_entity=True)
print(result.summary.tables[1])

mod = smf.ols("lwage ~ expersq+union+married+hours+black+hisp+educ", data=data).fit()
print(mod.summary().tables[1])

toy_panel = pd.DataFrame({
    "mkt_costs":[5,4,3.5,3, 10,9.5,9,8, 4,3,2,1, 8,7,6,4],
    "purchase":[12,9,7.5,7, 9,7,6.5,5, 15,14.5,14,13, 11,9.5,8,5],
    "city":["C0","C0","C0","C0", "C2","C2","C2","C2", "C1","C1","C1","C1", "C3","C3","C3","C3"]
})


m = smf.ols("purchase ~ mkt_costs", data=toy_panel).fit()
plt.scatter(toy_panel.mkt_costs, toy_panel.purchase)
plt.plot(toy_panel.mkt_costs, m.fittedvalues, c="C5", label="Regression Line")
plt.xlabel("Marketing Costs (in 1000)")
plt.ylabel("In-app Purchase (in 1000)")
plt.title("Simple OLS Model")
plt.legend()
plt.savefig(png_path + "biased_down.png")
plt.close()
# marketing department tends to spend more to place billboards on cities where the purchase level is lower

# run a fixed effect model, adding the city's indicator as a dummy variable to your model.
fe = smf.ols("purchase ~ mkt_costs + C(city)", data=toy_panel).fit()

fe_toy = toy_panel.assign(y_hat = fe.fittedvalues)

plt.scatter(toy_panel.mkt_costs, toy_panel.purchase, c=toy_panel.city)
for city in fe_toy["city"].unique():
    plot_df = fe_toy.query(f"city=='{city}'")
    plt.plot(plot_df.mkt_costs, plot_df.y_hat, c="C5")
plt.title("Fixed Effect Model")
plt.xlabel("Marketing Costs (in 1000)")
plt.ylabel("In-app Purchase (in 1000)")
plt.savefig(png_path + "fixed_effect_city.png")
plt.close()

# time effects
# adding a time dummy would control for variables that are fixed for each time period, but that might change across time
# One example of such a variable is inflation. 
# Prices and salary tend to go up with time, but the inflation on each time period is the same for all entities. 
# suppose that marriage is increasing with time. If the wage and marriage proportion also changes with time, 
# we would have time as a confounder. Since inflation also makes salary increase with time, 
# some of the positive association we see between marriage and wage would be simply because 
# both are increasing with time. To correct for that, we can add a dummy variable for each time period.

mod = PanelOLS.from_formula("lwage ~ expersq+union+married+hours+EntityEffects+TimeEffects",
                            data=data.set_index(["nr", "year"]))

result = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
print(result.summary.tables[1])

# When panel data won't help you:
# The most obvious one is when you have confounders that are changing in time. 
# Fixed effects can only eliminate bias from attributes that are constant for each individual.
# Another less obvious case when fixed effect fails is when you have reversed causality
