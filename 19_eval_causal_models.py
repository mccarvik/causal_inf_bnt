"""
script for ch 19
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from toolz import curry

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")

png_path = "pngs/ch19/"

# it isn't obvious at all how we achieve anything like a train-test paradigm in the case of causal inference. 
# That's because causal inference is interested in estimating an unobservable quantity
# if we can't see it, how the hell are we supposed to know if our models are any good at estimating it? 
# It is as if every entity had an underlying responsiveness, denoted by the slope of the line from treatment to outcome, but we can't measure it.

# The trick is to use aggregate measurements of elasticity. 
# Even if you can't estimate elasticity individually, you can do it for a group and that is what we will leverage here.

# As we'll see, random data is very valuable for evaluation purposes. 
# In real life, it is often expensive to collect random data 
# (why would you set prices at random if you know some of them are not very good ones and will only make you lose money???)
# What ends up happening is that we often have an abundance of data where the treatment is NOT random and very few, if any, random data
# Since evaluating a model with non random data is incredibly tricky, if we have any random data, we tend to leave that for evaluation purposes

prices = pd.read_csv("./data/ice_cream_sales.csv") # loads non-random data
prices_rnd = pd.read_csv("./data/ice_cream_sales_rnd.csv") # loads random data
print(prices_rnd.shape)
print(prices.head())

# linreg
m1 = smf.ols("sales ~ price*cost + price*C(weekday) + price*temp", data=prices).fit()

X = ["temp", "weekday", "cost", "price"]
y = "sales"
np.random.seed(1)
m2 = GradientBoostingRegressor()
# grad boosting - non parametric
m2.fit(prices[X], prices[y])

# some drop in performance is expected, because there is a concept drift. 
# The model was trained in data where price is not random, but the test set has only randomized prices
print("Train Score:", m2.score(prices[X], prices[y]))
print("Test Score:", m2.score(prices_rnd[X], prices_rnd[y]))

def predict_elast(model, price_df, h=0.01):
    return (model.predict(price_df.assign(price=price_df["price"]+h))
            - model.predict(price_df)) / h

np.random.seed(123)
prices_rnd_pred = prices_rnd.assign(**{
    "elast_m_pred": predict_elast(m1, prices_rnd), ## elasticity model
    "pred_m_pred": m2.predict(prices_rnd[X]), ## predictive model
    "rand_m_pred": np.random.uniform(size=prices_rnd.shape[0]), ## random model
})
print(prices_rnd_pred.head())


# model 1 is predicting highest sales
# model 2 is predicting highest elasticity
# model rand is random
# it would be very useful if we could somehow order units from more sensitive to less sensitive. 
# Since we have predicted elasticity, we can order the units by that prediction and hope it also orders them by real elasticity
# Sadly, we can't evaluate that ordering on a unit level. But, What if, instead, we evaluate groups defined by the ordering? 
# If our treatment is randomly distributed (and here is where randomness enters), 
# estimating elasticity for a group of units is easy.
# All we need is to compare the outcome between the treated and untreated.

# result:
# m1 will predict sales well but not differentiate treatment effects of the two groups
# m2 will predict sales and treatment effects
# rand will predict neither

@curry
def elast(data, y, t):
        # line coeficient for the one variable linear regression 
        return (np.sum((data[t] - data[t].mean())*(data[y] - data[y].mean())) /
                np.sum((data[t] - data[t].mean())**2))

# calculated elasticity by band
def elast_by_band(df, pred, y, t, bands=10):
    return (df
            .assign(**{f"{pred}_band":pd.qcut(df[pred], q=bands)}) # makes quantile partitions
            .groupby(f"{pred}_band")
            .apply(elast(y=y, t=t))) # estimate the elasticity on each partition

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 4))
for m, ax in zip(["elast_m_pred", "pred_m_pred", "rand_m_pred"], axs):
    elast_by_band(prices_rnd_pred, m, "sales", "price").plot.bar(ax=ax)
plt.savefig(png_path+"elasticities.png")
plt.close()

# random model (rand_m): It has roughly the same estimated elasticity in each of its partitions
# We can already see just by looking at the plot that it won't help us much with personalisation 
# since it can't distinguish between the high and low price elasticity days
# 
# m_pred: manages to construct groups where the elasticity is high and others where the elasticity is low. 
# 
# causal model elast_m: looks a bit weird. 
# It identifies groups of really low elasticity, where low here actually means high price sensitivity 
# (sales will decrease by a lot as we increase prices). 
# it can successfully distinguish high from low elasticities. 
# But the ordering is not as good as that of the predictive model.

# The predictive model has better ordering, but the causal model can better identify the extremes. 
# The elasticity by band plot is a good first check, but it can't answer precisely which model is better

def cumulative_elast_curve(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    
    # orders the dataset by the `prediction` column
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    
    # create a sequence of row numbers that will define our Ks
    # The last item is the sequence is all the rows (the size of the dataset)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    # cumulative computes the elasticity. First for the top min_periods units.
    # then for the top (min_periods + step*1), then (min_periods + step*2) and so on
    return np.array([elast(ordered_df.head(rows), y, t) for rows in n_rows])

# Cumulative Elasticity Curve: We first compute the elasticity of the first group
# then, of the first and the second and so on, until we've included all the groups
# first bin in the cumulative elasticity is just the ATE from the most sensitive group according to that model, and so on
# In words, if a model is good at ordering elasticity, 
# the elasticity observed in the top k samples should be higher than the elasticity observed in top k+a samples. 
# Or, simply put, if I look at the top units, they should have higher elasticity than units below them.
# The intuition being that not only do we want the elasticity of the top k units to be higher than the elasticity of the units below them, 
# but we want that difference to be as large as possible.

plt.figure(figsize=(10,6))

for m in ["elast_m_pred", "pred_m_pred", "rand_m_pred"]:
    cumu_elast = cumulative_elast_curve(prices_rnd_pred, m, "sales", "price", min_periods=100, steps=100)
    x = np.array(range(len(cumu_elast)))
    plt.plot(x/x.max(), cumu_elast, label=m)

plt.hlines(elast(prices_rnd_pred, "sales", "price"), 0, 1, linestyles="--", color="black", label="Avg. Elast.")
plt.xlabel("% of Top Elast. Days")
plt.ylabel("Cumulative Elasticity")
plt.title("Cumulative Elasticity Curve")
plt.legend()
plt.savefig(png_path+"cum_elast.png")
plt.close()

# Cumulative Gain Curve
# We will multiply the cumulative elasticity by the proportional sample size. 
# For example, if the cumulative elasticity is, say -0.5 at 40%, we will end up with -0.2 (-0.5 * 0.4) at that point. 
# Then, we will compare this with the theoretical curve produced by a random model. 
# This curve will actually be a straight line from 0 to the average treatment effect.
# All curves will start and end at the same point. However, the better the model at ordering elasticity, 
# the more the curve will diverge from the random line in the points between zero and one
# can think about Cumulative Gain as the ROC for causal models.

def cumulative_gain(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    ## add (rows/size) as a normalizer. 
    return np.array([elast(ordered_df.head(rows), y, t) * (rows/size) for rows in n_rows])

plt.figure(figsize=(10,6))

for m in ["elast_m_pred", "pred_m_pred", "rand_m_pred"]:
    cumu_gain = cumulative_gain(prices_rnd_pred, m, "sales", "price", min_periods=50, steps=100)
    x = np.array(range(len(cumu_gain)))
    plt.plot(x/x.max(), cumu_gain, label=m)
    
plt.plot([0, 1], [0, elast(prices_rnd_pred, "sales", "price")], linestyle="--", label="Random Model", color="black")

plt.xlabel("% of Top Elast. Days")
plt.ylabel("Cumulative Gain")
plt.title("Cumulative Gain")
plt.legend()
plt.savefig(png_path+"cum_gain.png")
plt.close()


# adding confidence intervals to cum_elast
def elast_ci(df, y, t, z=1.96):
    n = df.shape[0]
    t_bar = df[t].mean()
    beta1 = elast(df, y, t)
    beta0 = df[y].mean() - beta1 * t_bar
    e = df[y] - (beta0 + beta1*df[t])
    se = np.sqrt(((1/(n-2))*np.sum(e**2))/np.sum((df[t]-t_bar)**2))
    return np.array([beta1 - z*se, beta1 + z*se])

def cumulative_elast_curve_ci(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    # just replacing a call to `elast` by a call to `elast_ci`
    return np.array([elast_ci(ordered_df.head(rows), y, t)  for rows in n_rows])

plt.figure(figsize=(10,6))

cumu_gain_ci = cumulative_elast_curve_ci(prices_rnd_pred, "elast_m_pred", "sales", "price", min_periods=50, steps=200)
x = np.array(range(len(cumu_gain_ci)))
plt.plot(x/x.max(), cumu_gain_ci, color="C0")

plt.hlines(elast(prices_rnd_pred, "sales", "price"), 0, 1, linestyles="--", color="black", label="Avg. Elast.")

plt.xlabel("% of Top Elast. Days")
plt.ylabel("Cumulative Elasticity")
plt.title("Cumulative Elasticity for elast_m_pred with 95% CI")
plt.legend()
plt.savefig(png_path+"conf_ints_cum_elast.png")
plt.close()


# adding confidence intervals to cum_gain
def cumulative_gain_ci(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast_ci(ordered_df.head(rows), y, t) * (rows/size) for rows in n_rows])

plt.figure(figsize=(10,6))

cumu_gain = cumulative_gain_ci(prices_rnd_pred, "elast_m_pred", "sales", "price", min_periods=50, steps=200)
x = np.array(range(len(cumu_gain)))
plt.plot(x/x.max(), cumu_gain, color="C0")

plt.plot([0, 1], [0, elast(prices_rnd_pred, "sales", "price")], linestyle="--", label="Random Model", color="black")

plt.xlabel("% of Top Elast. Days")
plt.ylabel("Cumulative Gain")
plt.title("Cumulative Gain for elast_m_pred with 95% CI")
plt.legend()
plt.savefig(png_path+"conf_ints_cum_gain.png")
plt.close()
