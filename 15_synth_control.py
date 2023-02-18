"""
Script for ch15
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

pd.set_option("display.max_columns", 6)
style.use("fivethirtyeight")

png_path = "pngs/ch15/"

# We don't need to find any single unit in the untreated that is very similar to the treated. 
# Instead, we can forge our own as a combination of multiple untreated units, creating what is effectively a synthetic control. 

cigar = (pd.read_csv("data/smoking.csv")
         .drop(columns=["lnincome","beer", "age15to24"]))
print(cigar.query("california").head())

ax = plt.subplot(1, 1, 1)

(cigar
 .assign(california = np.where(cigar["california"], "California", "Other States"))
 .groupby(["year", "california"])
 ["cigsale"]
 .mean()
 .reset_index()
 .pivot("year", "california", "cigsale")
 .plot(ax=ax, figsize=(10,5)))

plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Cigarette Sales Trend")
plt.title("Gap in per-capita cigarette sales (in packs)")
plt.legend()
plt.savefig(png_path + "cig_sales.png")
plt.close()

# To answer the question of whether Proposition 99 had an effect on cigarette consumption, 
# Will use the pre-intervention period to build a synthetic control. 
# We will combine the other states to build a fake state that resembles very closely the trend of California. 
# Then, we will see how this synthetic control behaves after the intervention.

# We will build a synthetic control that looks a lot like California in the pre intervention period and 
# see how it would behave in the post intervention period.
features = ["cigsale", "retprice"]
inverted = (cigar.query("~after_treatment") # filter pre-intervention period
            .pivot(index='state', columns="year")[features] # make one column per year and one row per state
            .T) # flip the table to have one column per state
print(inverted.head())

# Now, we can define our Y variable as the state of California and the X as the other states.
y = inverted[3].values # state of california
X = inverted.drop(columns=3).values  # other states
from sklearn.linear_model import LinearRegression
weights_lr = LinearRegression(fit_intercept=False).fit(X, y).coef_
# Then, we run a regression. The regression will return the set of weights that minimize the square difference between 
# the treated unit and the units in the donor pool.
print(weights_lr.round(3))

# These weights show us how to build the synthetic control. 
# We will multiply the outcome of state 1 by -0.436, of state 2 by -1.038, of state 4 by 0.679 and so on. 
# We can achieve this with a dot product between the matrix from the states in the pool and the weights.
calif_synth_lr = (cigar.query("~california")
                  .pivot(index='year', columns="state")["cigsale"]
                  .values.dot(weights_lr))
# Now that we have our synthetic control, we can plot it with the outcome variable of the State of California.
plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"], label="California")
plt.plot(cigar.query("california")["year"], calif_synth_lr, label="Synthetic Control")
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend()
plt.savefig(png_path + "synth_cont_off.png")
plt.close()
# notice how the pre-intervention period is fitted perfectly. 
# The synthetic control is able to match the state of California exactly. 
# This is a sign that our synthetic control model is probably overfitting the data.
# Another sign is the huge variance on the outcome variable of the synthetic control after the intervention. 
# Notice how it doesnt follow smooth patterns. Instead, it goes up and down and up and down.

# Remember that we have 38 states in our donor pool. 
# So our linear regression has 38 parameters to play with in order to make the pretreatment pool 
# match the treatment as close as it can. This is the case where, even if T is large, N is also large, 
# which gives too much flexibility to our linear regression model. 
# If you are familiar with regularized models, know that you could use Ridge or Lasso regression to fix this.
# Here, we will look at another more traditional way to avoid overfitting.

from typing import List
from operator import add
from toolz import reduce, partial

def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W))**2))

from scipy.optimize import fmin_slsqp

def get_w(X, y):
    
    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights

calif_weights = get_w(X, y)
print("Sum:", calif_weights.sum())
print(np.round(calif_weights, 4))

# Regularization technique - Interpolation, not extrapolation
# With this weight, we are multiplying states 1,2, and 3 by zero, state 4 by 0.0852 and so on. 
# Notice how the weights are sparse, exactly as we've predicted. 
# Also, all weights sum to one and are between 0 and 1, satisfying our convex combination constraint.
# Now, to get the synthetic control, we can multiply those weights by the states exactly as we did before with the regression weights.
calif_synth = cigar.query("~california").pivot(index='year', columns="state")["cigsale"].values.dot(calif_weights)
# If we plot the outcome of the synthetic control now, we get a much smoother trend. 
# Also notice that, in the pre intervention period, the synthetic control doesn't reproduce the treated exactly anymore. 
# This is a good sign, as it indicates that we are not overfitting.
plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"], label="California")
plt.plot(cigar.query("california")["year"], calif_synth, label="Synthetic Control")
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Per-capita cigarette sales (in packs)")
plt.legend()
plt.savefig(png_path + "new_synth.png")
plt.close()


plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth,
         label="California Effect")
plt.vlines(x=1988, ymin=-30, ymax=7, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2)
plt.title("State - Synthetic Across Time")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend()
plt.savefig(png_path + "synth_eff.png")
plt.close()

# how can I know if this is statistically significant?
# Since our sample size is very small (39), we will have to be a bit smarter when figuring out if our result is statistically significant 
# and not just due to random luck. Here, we will use the idea of Fisher's Exact Test. We permute the treated and control exhaustively. 
# Since we only have one treated unit, this would mean that, for each unit, we pretend it is the treated while the others are the control.
# In the end, we will have one synthetic control and effect estimates for each state. 
# So what this does is it pretends that the treatment actually happened for another state, not California, and 
# see what would have been the estimated effect for this treatment that didn't happen. 
# Then, we see if the treatment in California is sufficiently larger when compared to the other fake treatment. 
# The idea is that for states that weren't actually treated, once we pretend they were, 
# we won't be able to find any significant treatment effect.

def synthetic_control(state: int, data: pd.DataFrame) -> np.array:
    features = ["cigsale", "retprice"]
    
    inverted = (data.query("~after_treatment")
                .pivot(index='state', columns="year")[features]
                .T)
    
    y = inverted[state].values # treated
    X = inverted.drop(columns=state).values # donor pool

    weights = get_w(X, y)
    synthetic = (data.query(f"~(state=={state})")
                 .pivot(index='year', columns="state")["cigsale"]
                 .values.dot(weights))
    return (data
            .query(f"state=={state}")[["state", "year", "cigsale", "after_treatment"]]
            .assign(synthetic=synthetic))
print(synthetic_control(1, cigar).head())


from joblib import Parallel, delayed
control_pool = cigar["state"].unique()
parallel_fn = delayed(partial(synthetic_control, data=cigar))
synthetic_states = Parallel(n_jobs=8)(parallel_fn(state) for state in control_pool)
print(synthetic_states[0].head())

# With the synthetic control for all the states, we can estimate the gap between the synthetic and the true state for all states.
# For California, this is the treatment effect. For the other states, this is like a placebo effect, 
# where we estimate the synthetic control treatment effect where the treatment didn't actually happen. 
# If we plot all the placebo effects along with the California treatment effect, we get the following figure.

plt.figure(figsize=(12,7))
for state in synthetic_states:
    plt.plot(state["year"], state["cigsale"] - state["synthetic"], color="C5",alpha=0.4)

plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth,
        label="California");

plt.vlines(x=1988, ymin=-50, ymax=120, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("State - Synthetic Across Time")
plt.legend()
plt.savefig(png_path + "synth_eff_time.png")
plt.close()


# Remove states that dont fit California well
def pre_treatment_error(state):
    pre_treat_error = (state.query("~after_treatment")["cigsale"] 
                       - state.query("~after_treatment")["synthetic"]) ** 2
    return pre_treat_error.mean()

plt.figure(figsize=(12,7))
for state in synthetic_states:
    # remove units with mean error above 80.
    if pre_treatment_error(state) < 80:
        plt.plot(state["year"], state["cigsale"] - state["synthetic"], color="C5",alpha=0.4)

plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth,
        label="California");

plt.vlines(x=1988, ymin=-50, ymax=120, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("Distribution of Effects")
plt.title("State - Synthetic Across Time (Large Pre-Treatment Errors Removed)")
plt.legend()
plt.savefig(png_path+"remove_outlier_states.png")
plt.close()
# Removing the noise, we can see how extreme of a value is the effect in the state of California. 
# This image shows us that if we pretend the treatment had happened to any other state, 
# we would almost never get an effect so extreme as the one we got with California.

# This picture alone is a form of inference, but we can also derive a P-value from these results. 
# All we have to do is see how many times the effects that we've got is below the effect of California.

calif_number = 3
effects = [state.query("year==2000").iloc[0]["cigsale"] - state.query("year==2000").iloc[0]["synthetic"]
           for state in synthetic_states
           if pre_treatment_error(state) < 80] # filter out noise

calif_effect = cigar.query("california & year==2000").iloc[0]["cigsale"] - calif_synth[-1] 
print("California Treatment Effect for the Year 2000:", calif_effect)
np.array(effects)

# only 1/35 other states have a result as good as Californias
print(np.mean(np.array(effects) < calif_effect))

# distribution
_, bins, _ = plt.hist(effects, bins=20, color="C5", alpha=0.5);
plt.hist([calif_effect], bins=bins, color="C0", label="California")
plt.ylabel("Frquency")
plt.title("Distribution of Effects")
plt.legend()
plt.savefig(png_path+"dist.png")
plt.close()
