"""
script for ch16
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

style.use("fivethirtyeight")

png_path = "pngs/ch16/"

drinking = pd.read_csv("./data/drinking.csv")
print(drinking.head()[["agecell", "all", "mva", "suicide"]])
drinking["agecell"] -= 21

plt.figure(figsize=(8,8))
ax = plt.subplot(3,1,1)
drinking.plot.scatter(x="agecell", y="all", ax=ax)
plt.title("Death Cause by Age (Centered at 0)")

ax = plt.subplot(3,1,2, sharex=ax)
drinking.plot.scatter(x="agecell", y="mva", ax=ax)

ax = plt.subplot(3,1,3, sharex=ax)
drinking.plot.scatter(x="agecell", y="suicide", ax=ax);
plt.savefig(png_path + "agecell.png")
plt.close()

# To make it work, we interact a dummy for being above the threshold with the running variable
# Essentially, this is the same as fitting a linear regression above the threshold and another below it. 
# The parameter B0 is the intercept of the regression below the threshold and 
# The parameters B0 + B2 is the intercept for the regression above the threshold.
rdd_df = drinking.assign(threshold=(drinking["agecell"] > 0).astype(int))
model = smf.wls("all~agecell*threshold", rdd_df).fit()
print(model.summary().tables[1])

# As if we had two regression models
ax = drinking.plot.scatter(x="agecell", y="all", color="C0")
drinking.assign(predictions=model.fittedvalues).plot(x="agecell", y="predictions", ax=ax, color="C1")
plt.title("Regression Discontinuity")
plt.savefig(png_path + "2_regr.png")
plt.close()


plt.figure(figsize=(8,8))
for p, cause in enumerate(["all", "mva", "suicide"], 1):
    ax = plt.subplot(3,1,p)
    drinking.plot.scatter(x="agecell", y=cause, ax=ax)
    m = smf.wls(f"{cause}~agecell*threshold", rdd_df).fit()
    ate_pct = 100*((m.params["threshold"] + m.params["Intercept"])/m.params["Intercept"] - 1)
    drinking.assign(predictions=m.fittedvalues).plot(x="agecell", y="predictions", ax=ax, color="C1")
    plt.title(f"Impact of Alcohol on Death: {np.round(ate_pct, 2)}%")

plt.tight_layout()
plt.savefig(png_path + "all_causes.png")
plt.close()


# Kernel Weighting
# Regression Discontinuity relies heavily on the extrapolations properties of linear regression. 
# Since we are looking at the values at the beginning and end of 2 regression lines, we better get those limits right. 
# What can happen is that regression might focus too much on fitting the other data points at the cost of a poor fit at the threshold. 
# If this happens, we might get the wrong measure of the treatment effect.
# One way to solve this is to give higher weights for the points that are closer to the threshold. 
# One is to reweight the samples with the triangular kernel

def kernel(R, c, h):
    indicator = (np.abs(R-c) <= h).astype(float)
    return indicator * (1 - np.abs(R-c)/h)
plt.plot(drinking["agecell"], kernel(drinking["agecell"], c=0, h=1))
plt.xlabel("agecell")
plt.ylabel("Weight")
plt.title("Kernel Weight by Age");
plt.savefig(png_path+"kernel_wgt.png")
plt.close()

model = smf.wls("all~agecell*threshold", rdd_df,
                weights=kernel(drinking["agecell"], c=0, h=1)).fit()
print(model.summary().tables[1])

ax = drinking.plot.scatter(x="agecell", y="all", color="C0")
drinking.assign(predictions=model.fittedvalues).plot(x="agecell", y="predictions", ax=ax, color="C1")
plt.title("Regression Discontinuity (Local Regression)");
plt.savefig(png_path + "regr_disc.png")
plt.close()

plt.figure(figsize=(8,8))
weights = kernel(drinking["agecell"], c=0, h=1)
for p, cause in enumerate(["all", "mva", "suicide"], 1):
    ax = plt.subplot(3,1,p)
    drinking.plot.scatter(x="agecell", y=cause, ax=ax)
    m = smf.wls(f"{cause}~agecell*threshold", rdd_df, weights=weights).fit()
    ate_pct = 100*((m.params["threshold"] + m.params["Intercept"])/m.params["Intercept"] - 1)
    drinking.assign(predictions=m.fittedvalues).plot(x="agecell", y="predictions", ax=ax, color="C1")
    plt.title(f"Impact of Alcohol on Death: {np.round(ate_pct, 2)}%")
plt.tight_layout()
plt.savefig(png_path + "regr_disc_all.png")
plt.close()

# sheepskin effect
# In order to graduate in Texas, one has to pass an exam. 
# get data from students who took those last chance exams and compare those that had barely failed it to those that barely passed
# These students will have very similar human capital, but different signaling credentials. 
# Namely, those that barely passed it, will receive a diploma.

sheepskin = pd.read_csv("./data/sheepskin.csv")[["avgearnings", "minscore", "receivehsd", "n"]]
print(sheepskin.head())


# some slippage in the treatment assignment. 
# Some students that are below the passing threshold managed to get the diploma anyway. 
# Here, the regression discontinuity is fuzzy, rather than sharp. 
# Notice how the probability of getting the diploma doesn't jump from zero to one at the threshold.
# But it does jump from something like 50% to 90%.
sheepskin.plot.scatter(x="minscore", y="receivehsd", figsize=(10,5))
plt.xlabel("Test Scores Relative to Cut off")
plt.ylabel("Fraction Receiving Diplomas")
plt.title("Last-chance Exams");
plt.savefig(png_path+"sheep_dipl.png")
plt.close()

# One thing that could break our RDD argument is if people can manipulate where they stand at the threshold. 
# if students just below the threshold found a way around the system to increase their test score by just a bit. 
# Another example is when you need to be below a certain income level to get a government benefit. 
# Some families might lower their income on purpose, just to be just eligible for the program.

# In these sorts of situations, we tend to see a phenomenon called bunching on the density of the running variable. 
# This means that we will have a lot of entities just above or just below the threshold. 
# To check for that, we can plot the density function of the running variable and see if there are any spikes around the threshold. 
# For our case, the density is given by the n column in our data.

plt.figure(figsize=(8,8))

ax = plt.subplot(2,1,1)
sheepskin.plot.bar(x="minscore", y="n", ax=ax)
plt.title("McCrary Test")
plt.ylabel("Smoothness at the Threshold")

ax = plt.subplot(2,1,2, sharex=ax)
sheepskin.replace({1877:1977, 1874:2277}).plot.bar(x="minscore", y="n", ax=ax)
plt.xlabel("Test Scores Relative to Cut off")
plt.ylabel("Spike at the Threshold")
plt.savefig(png_path+"bunching.png")
plt.close()

sheepsking_rdd = sheepskin.assign(threshold=(sheepskin["minscore"]>0).astype(int))
model = smf.wls("avgearnings~minscore*threshold",
                sheepsking_rdd,
                weights=kernel(sheepsking_rdd["minscore"], c=0, h=15)*sheepsking_rdd["n"]).fit()
print(model.summary().tables[1])

ax = sheepskin.plot.scatter(x="minscore", y="avgearnings", color="C0")
sheepskin.assign(predictions=model.fittedvalues).plot(x="minscore", y="predictions", ax=ax, color="C1", figsize=(8,5))
plt.xlabel("Test Scores Relative to Cutoff")
plt.ylabel("Average Earnings")
plt.title("Last-chance Exams")
plt.savefig(png_path+"last_chance.png")
plt.close()

# Need to check for noncompliance
def wald_rdd(data):
    weights=kernel(data["minscore"], c=0, h=15)*data["n"]
    denominator = smf.wls("receivehsd~minscore*threshold", data, weights=weights).fit()
    numerator = smf.wls("avgearnings~minscore*threshold", data, weights=weights).fit()
    return numerator.params["threshold"]/denominator.params["threshold"]
from joblib import Parallel, delayed 

np.random.seed(45)
bootstrap_sample = 1000
ates = Parallel(n_jobs=4)(delayed(wald_rdd)(sheepsking_rdd.sample(frac=1, replace=True))
                          for _ in range(bootstrap_sample))
ates = np.array(ates)

sns.distplot(ates, kde=False)
plt.vlines(np.percentile(ates, 2.5), 0, 100, linestyles="dotted")
plt.vlines(np.percentile(ates, 97.5), 0, 100, linestyles="dotted", label="95% CI")
plt.title("ATE Bootstrap Distribution")
plt.xlim([-10000, 10000])
plt.legend()
plt.savefig(png_path+"still_not_sig.png")
plt.close()

