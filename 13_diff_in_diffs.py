"""
script for ch13
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

png_path = "pngs/ch13/"

style.use("fivethirtyeight")
data = pd.read_csv("data/billboard_impact.csv")
print(data.head())

#  Diff-in-diff is commonly used to assess the effect of macro interventions, 
# like the effect of immigration on unemployment, the effect of gun law changes in crime rates or 
# simply the difference in user engagement due to a marketing campaign. 
# In all these cases, you have a period before and after the intervention and 
# you wish to untangle the impact of the intervention from a general trend. 

poa_before = data.query("poa==1 & jul==0")["deposits"].mean()
poa_after = data.query("poa==1 & jul==1")["deposits"].mean()
print(poa_after - poa_before)
# This estimator is telling us that we should expect deposits to increase R$ 41,04 after the intervention
# This is false tho
# It is saying that in the case of no intervention, the outcome in the latter period would be the same as the outcome 
# from the starting period. This would obviously be false if your outcome variable follows any kind of trend.
# if deposits are going up -> the outcome of the latter period would be greater than that of the starting period even without intervention

# Another idea is to compare the treated group with an untreated group that didn't get the intervention:
fl_after = data.query("poa==0 & jul==1")["deposits"].mean()
print(poa_after - fl_after)
# this would only be true if both groups have a very similar baseline level

# this is not a great idea. To solve this, we can use both space and time comparison. 
# This is the idea of the difference in difference approach

# What this does is take the treated unit before the intervention and adds a trend component to it, 
# which is estimated using a control

# it is saying that the treated after the intervention, had it not been treated, 
# would look like the treated before the treatment plus a growth factor that is the same as the growth of the control
# notice that this assumes that the trends in the treatment and control are the same

# we get the classical Diff-in-Diff estimator
# name: because it gets the difference between the difference between treatment and control after and before the treatment.
fl_before = data.query("poa==0 & jul==0")["deposits"].mean()
diff_in_diff = (poa_after-poa_before)-(fl_after-fl_before)
print(diff_in_diff)

plt.figure(figsize=(10,5))
plt.plot(["May", "Jul"], [fl_before, fl_after], label="FL", lw=2)
plt.plot(["May", "Jul"], [poa_before, poa_after], label="POA", lw=2)

plt.plot(["May", "Jul"], [poa_before, poa_before+(fl_after-fl_before)],
         label="Counterfactual", lw=2, color="C2", ls="-.")
plt.legend()
plt.savefig(png_path + "diff_diff.png")
plt.close()

print(smf.ols('deposits ~ poa*jul', data=data).fit().summary().tables[1])

# One obvious problem with Diff-in-Diff is failure to satisfy the parallel trend assumption. 
# If the growth trend from the treated is different from the trend of the control, diff-in-diff will be biased.

# One way to check if this is happening is to plot the trend using past periods. 
# For example, let's suppose POA had a small decreasing trend but Florianopolis was on a steep ascent. 
# In this case, showing periods from before would reveal those trends and 
# we would know Diff-in-Diff is not a reliable estimator.
plt.figure(figsize=(10,5))
x = ["Jan", "Mar", "May", "Jul"]
plt.plot(x, [120, 150, fl_before,  fl_after], label="FL", lw=2)
plt.plot(x, [60, 50, poa_before, poa_after], label="POA", lw=2)
plt.plot(["May", "Jul"], [poa_before, poa_before+(fl_after-fl_before)], label="Counterfactual", lw=2, color="C2", ls="-.")
plt.legend()
plt.savefig(png_path + "non_para_trend.png")
plt.close()


