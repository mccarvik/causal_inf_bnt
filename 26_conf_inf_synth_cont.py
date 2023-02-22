"""
script for ch26
"""

import numpy as np
import pandas as pd
from toolz import curry
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import cvxpy as cp

import toolz as f

from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')

from matplotlib import style
style.use("ggplot")

png_path = "pngs/ch26/"

# Synthetic Control Refresher
# Synthetic Control (SC) is a particularly useful causal inference technique for when you have a single treatment unit and 
# very few control units, but you have repeated observation of each unit through time 
# Synthetic Controls tries to model Y(0) for the treated unit by combining multiple control units in such a way that 
# they mimic the pre-treatment behavior of the treated unit

# n our case, this means finding a combination of states that, together, 
# approximate the cigarette sales trend in California prior to Proposition 99.
# We can see that by plotting the cigarette sales trend for multiple states. 
# Notice none of them have a trend that closely resembles that of California.

data = pd.read_csv("data/smoking.csv")
data = data.pivot("year", "state", "cigsale")
data = data.rename(columns={c: f"state_{c}" for c in data.columns}).rename(columns={"state_3": "california"})
print(data.shape)
plt.figure(figsize=(10,5))
plt.plot(data.drop(columns=["california"]), color="C1", alpha=0.5)
plt.plot(data["california"], color="C0", label="California")
plt.vlines(x=1988, ymin=40, ymax=300, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.legend()
plt.ylabel("Cigarette Sales")
plt.savefig(png_path+"cig_sales.png")
plt.close()

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp

class SyntheticControl(BaseEstimator, RegressorMixin):

    def __init__(self,):
        pass

    def fit(self, X, y):

        X, y = check_X_y(X, y)
    
        w = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X@w - y))
        
        constraints = [cp.sum(w) == 1, w >= 0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        self.X_ = X
        self.y_ = y
        self.w_ = w.value
        
        self.is_fitted_ = True
        return self
        
        
    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)
        return X @ self.w_


model = SyntheticControl()
train = data[data.index < 1988]
model.fit(train.drop(columns=["california"]), train["california"])
plt.plot(data["california"], label="California")
plt.plot(data["california"].index, model.predict(data.drop(columns=["california"])), label="SC")
plt.vlines(x=1988, ymin=40, ymax=120, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.legend()
plt.savefig(png_path+"synth_cont.png")
plt.close()
# plot side by side the trend for California and for the synthetic control we've just created. 
# The difference between these two lines is the estimated effect of Proposition 99 in California.
# From the look of this plot, it looks like Proposition 99 had a pretty big effect on the reduction of cigarette sales.

pred_data = data.assign(**{"residuals": data["california"] - model.predict(data.drop(columns=["california"]))})
plt.plot(pred_data["california"].index, pred_data["residuals"], label="Estimated Effect")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2, color="Black")
plt.vlines(x=1988, ymin=5, ymax=-25, linestyle=":", lw=2, label="Proposition 99", color="Black")
plt.legend()
plt.savefig(png_path+"diff_for_cali.png")
plt.close()

plt.figure(figsize=(10,5))
for state in data.columns:
    
    model_ier = SyntheticControl()
    train_iter = data[data.index < 1988]
    model_ier.fit(train_iter.drop(columns=[state]), train_iter[state])
    
    effect = data[state] - model_ier.predict(data.drop(columns=[state]))
    
    is_california = state == "california"
    
    plt.plot(effect,
             color="C0" if is_california else "C1",
             alpha=1 if is_california else 0.5,
             label="California" if is_california else None)

plt.hlines(y=0, xmin=1970, xmax=2000, lw=2, color="Black")
plt.vlines(x=1988, ymin=-50, ymax=100, linestyle=":", lw=2, label="Proposition 99", color="Black")
plt.ylabel("Effect Estimate")
plt.legend()
plt.savefig(png_path+"cali_effect.png")
plt.close()
# In our example, we can see that the post-treatment difference for California is quite extreme, when compared to the other states.

# Hypothesis Test and P-Values
# we wish to test for no effect whatsoeve
# The key idea is to then generate data following the null hypothesis we want to test and 
# check the residuals of a model for Y(0) in this generated data.
#  If the residuals are too extreme, we say that the data is unlikely to have come from the null hypothesis we've postulated

def with_effect(df, state, null_hypothesis, start_at, window):
    window_mask = (df.index >= start_at) & (df.index < (start_at +window))
    
    y = np.where(window_mask, df[state] - null_hypothesis, df[state])
    
    return df.assign(**{state: y})
plt.plot(with_effect(data, "california", 0, 1988, 2000-1988+1)["california"], label="H0: 0")
plt.plot(with_effect(data, "california", -4, 1988, 2000-1988+1)["california"], label="H0: -4")
plt.ylabel("Y0 Under the Null")
plt.legend()
plt.savefig(png_path+"null_hypo.png")
plt.close()

# The next part of the inference procedure is to fit a model for the counterfactual Y(0)
# (which we get with the function we just created) in the entire data, pre and post-treatment period

@curry
def residuals(df, state, null, intervention_start, window, model):
    
    null_data = with_effect(df, state, null, intervention_start, window)
            
    model.fit(null_data.drop(columns=[state]), null_data[state])
    
    y0_est = pd.Series(model.predict(null_data.drop(columns=[state])), index=null_data.index)
    
    residuals = null_data[state] - y0_est
    
    test_mask = (null_data.index >= intervention_start) & (null_data.index < (intervention_start + window))
    
    return pd.DataFrame({
        "y0": null_data[state],
        "y0_est": y0_est,
        "residuals": residuals,
        "post_intervention": test_mask
    })[lambda d: d.index < (intervention_start + window)]  # just discard  points after the defined effect window

# With our data, to get the residuals for H0 : 0, meaning Proposition 99 had no effect, we can simply pass 0 as the null for our function.
model = SyntheticControl()
residuals_df = residuals(data,
                         "california",
                         null=0.0,
                         intervention_start=1988,
                         window=2000-1988+1,
                         model=model)
print(residuals_df.head())

# The result is a dataframe containing the estimated residuals for each time period, something we will use going forward. 
# Remember that the idea here is to see if that residual, in the post intervention period, is too high. 
# If it is, the data is unlikely to have come from this null, where the effect is zero. 
# To get a visual idea of what we are talking about, we can inspect the error of our model in the post intervention period.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
residuals_df[["y0", "y0_est"]].plot(ax=ax1)
ax1.set_title("Y0 under H0: 0");
residuals_df[["residuals"]].plot(ax=ax2)
ax2.set_title("Residuals under H0: 0")
plt.savefig(png_path+"neg_hypth.png")
plt.close()
# We can already see that the model fitted under yields quite large and negative residuals, 
# which is some evidence we might want to reject this null of no effect.

# test statistic
# Test Statistic S, which summarizes how big are the residuals and hence, how unikly is the data we saw, under the null.
def test_statistic(u_hat, q=1, axis=0):
    return (np.abs(u_hat) ** q).mean(axis=axis) ** (1/q)

print("H0:0 ", test_statistic(residuals_df.query("post_intervention")["residuals"]))
# High values of this test statistic indicate poor post intervention fit and, hence rejection of the null
# To compute the P-value, we block-permute the residuals, calculating the test statistic in each permutation
# Once we do that, we will end up with T test statistics, one for each of the block permutations

# n plain terms, we are simply finding the proportion of times that the unpermuted test statistic is higher (more extreme) 
# than the test statistics obtained by all possible block permutations.
def p_value(resid_df, q=1):
    u = resid_df["residuals"].values
    post_intervention = resid_df["post_intervention"].values
    block_permutations = np.stack([np.roll(u, permutation, axis=0)[post_intervention]
                                   for permutation in range(len(u))])
    statistics = test_statistic(block_permutations, q=1, axis=1)
    p_val = np.mean(statistics >= statistics[0])
    return p_val

print(p_value(residuals_df))
# this is the P-value for the null hypothesis which states that the effect in all time periods is zero
# might be interesting to plot the confidence interval for effect each post treatment period individually, 
# rather than just testing a null hypothesis about an entire affect trajectory.

# Confidence Intervals
# To understand how we can place a confidence interval around the effect of each post-treatment period,
# let's first try to understand how we would define the confidence interval for a single time period. 
# If we have a single period, then H0 is defined in terms of a scalar value, rather than a trajectory vector
# This means we can generate a fine line of H0s and compute the P-value associated with each null. 
# For example, Let's say we think the effect of Proposition 99 in the year 1988 (the year it passed) 
# is somewhere between -20 and 20. We can then build a table containing a bunch of H0, from -20 to 20, and each associated P-value:

def p_val_grid(df, state, nulls, intervention_start, period, model):
    df_aug = df[df.index < intervention_start].append(df.loc[period])
    
    p_vals =  {null: p_value(residuals(df_aug,
                                       state,
                                       null=null,
                                       intervention_start=period,
                                       window=1,
                                       model=model)) for null in nulls}        
        
    return pd.DataFrame(p_vals, index=[period]).T

model = SyntheticControl()
nulls = np.linspace(-20, 20, 100)
p_values_df = p_val_grid(
    data,
    "california",
    nulls=nulls,
    intervention_start=1988,
    period=1988,
    model=model
)
print(p_values_df)


# To build the confidence interval, all we need to do is filter out the H0s that gave us a low P-value. 
# Remember that low p-value means that the data we have is unlikely to have come from that null. 
# For instance, if we define the significant level a to be 0.1, we remove H0s that have P-value lower than 0.1.
def confidence_interval_from_p_values(p_values, alpha=0.1):
    big_p_values = p_values[p_values.values >= alpha]
    return pd.DataFrame({
        f"{int(100-alpha*100)}_ci_lower": big_p_values.index.min(),
        f"{int(100-alpha*100)}_ci_upper": big_p_values.index.max(),
    }, index=[p_values.columns[0]])
print(confidence_interval_from_p_values(p_values_df))
# This gives us the confidence interval for the effect in 1988.

# We can also plot the H0 by P-value to better understand how this confidence interval was obtained
plt.plot(p_values_df[1988], p_values_df.index)
plt.xlabel("P-Value")
plt.ylabel("H0")
plt.vlines(0.1, nulls.min(), nulls.max(), color="black", ls="dotted", label="0.1")

plt.hlines(confidence_interval_from_p_values(p_values_df)["90_ci_upper"], 0, 1, color="C1", ls="dashed")
plt.hlines(confidence_interval_from_p_values(p_values_df)["90_ci_lower"], 0, 1, color="C1", ls="dashed", label="90% CI")

plt.legend()
plt.title("Confidence Interval for the Effect in 1988")
plt.savefig(png_path+"CI_pval.png")
plt.close()
# All there's left to do is repeat the procedure above for each time period.

def compute_period_ci(df, state, nulls, intervention_start, period, model, alpha=0.1):
    p_vals = p_val_grid(df=df,
                        state=state,
                        nulls=nulls,
                        intervention_start=intervention_start,
                        period=period,
                        model=model)
    
    return confidence_interval_from_p_values(p_vals, alpha=alpha)


def confidence_interval(df, state, nulls, intervention_start, window, model, alpha=0.1, jobs=4):    
    return pd.concat([compute_period_ci(df, state, nulls, intervention_start, period, model, alpha)
                     for period in range(intervention_start, intervention_start+window)])
# We are now ready to compute the confidence interval for all the post-intervention periods

model = SyntheticControl()
nulls = np.linspace(-60, 20, 100)
ci_df = confidence_interval(
    data,
    "california",
    nulls=nulls,
    intervention_start=1988,
    window=2000 - 1988 + 1,
    model=model
)
print(ci_df)
plt.figure(figsize=(10,5))
plt.fill_between(ci_df.index, ci_df["90_ci_lower"], ci_df["90_ci_upper"], alpha=0.2,  color="C1")
plt.plot(pred_data["california"].index, pred_data["residuals"], label="California", color="C1")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2, color="Black")
plt.vlines(x=1988, ymin=10, ymax=-50, linestyle=":", color="Black", lw=2, label="Proposition 99")
plt.legend()
plt.ylabel("Gap in per-capita cigarette sales (in packs)");
plt.savefig(png_path+"Gap_in_percapit.png")
plt.close()

