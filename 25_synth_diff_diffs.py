"""
script for ch25
"""
import pdb
import numpy as np
import pandas as pd
from toolz import curry, partial
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import cvxpy as cp

import warnings
warnings.filterwarnings('ignore')

from matplotlib import style
style.use("ggplot")
from joblib import Parallel, delayed # for parallel processing

png_path = "pngs/ch25/"

# In previous chapters, we looked into both Difference-in-Differences and Synthetic Control methods 
# for identifying the treatment effect with panel data (data where we have multiple units observed across multiple time periods). 
# It turns out we can merge both approaches into a single estimator. 
# This new Synthetic Difference-in-Differences estimation procedure manages to exploit advantages of both methods 
# while also increasing the precision (decreasing the error bars) of the treatment effect estimate.

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


pd.set_option('display.max_columns', 10)
data = (pd.read_csv("data/smoking.csv")[["state", "year", "cigsale", "california", "after_treatment"]]
        .rename(columns={"california": "treated"})
        .replace({"state": {3: "california"}}))
print(data.head())
print(data.query("state=='california'").query("year.between(1986, 1990)"))
data_piv = data.pivot("year", "state", "cigsale")
data_piv = data_piv.rename(columns={c: f"state_{c}" for c in data_piv.columns if c != "california"})
print(data_piv.head()[["state_1", "state_2", "state_4", "state_38", "state_39", "california"]].round())

did_model = smf.ols("cigsale ~ after_treatment*treated", data=data).fit()
att = did_model.params["after_treatment[T.True]:treated[T.True]"]
print("DiD ATT: ", att.round(3))

pre_year = data.query("~after_treatment")["year"].mean()
post_year = data.query("after_treatment")["year"].mean()

pre_control_y = did_model.params["Intercept"]
post_control_y = did_model.params["Intercept"] + did_model.params["after_treatment[T.True]"]

pre_treat_y = did_model.params["Intercept"] + did_model.params["treated[T.True]"]

post_treat_y0 = post_control_y + did_model.params["treated[T.True]"]

post_treat_y1 = post_treat_y0 + did_model.params["after_treatment[T.True]:treated[T.True]"]

plt.plot([pre_year, post_year], [pre_control_y, post_control_y], color="C0", label="Control")
plt.plot([pre_year, post_year], [pre_treat_y, post_treat_y0], color="C1", ls="dashed")
plt.plot([pre_year, post_year], [pre_treat_y, post_treat_y1], color="C1", label="California")

plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.title("DiD Estimation")
plt.ylabel("Cigarette Sales")
plt.legend()
plt.savefig(png_path+"did_est.png")
plt.close()
# This estimate should be taken with a grain of salt, though. 
# We know that Diff-in-Diff requires the trend in the control group to be equal to that of the treated group in the absence of the treatment
# Cigarette sales in California are decreasing faster than the average of the control states, even prior to the treatment. 
# If this trend extends beyond the pre-treatment period, the DiD estimator will be downward biased, meaning that the true effect is actually less extreme than the one we've estimated above.

plt.figure(figsize=(10,5))
plt.plot(data_piv.drop(columns=["california"]), color="C1", alpha=0.3)
plt.plot(data_piv.drop(columns=["california"]).mean(axis=1), lw=3, color="C1", ls="dashed", label="Control Avg.")
plt.plot(data_piv["california"], color="C0", label="California")
plt.vlines(x=1988, ymin=40, ymax=300, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.legend()
plt.ylabel("Cigarette Sales")
plt.title("Non-Parallel Trends")
plt.savefig(png_path+"nonpara_trends.png")
plt.close()
# The problem of non-parallel trends is where Synthetic Control comes into play in the Synthetic Diff-in-Diff model

@curry
def demean(df, col_to_demean):
    return df.assign(**{col_to_demean: (df[col_to_demean]
                                        - df.groupby("state")[col_to_demean].transform("mean")
                                        - df.groupby("year")[col_to_demean].transform("mean"))})

formula = f"""cigsale ~ treat"""
mod = smf.ols(formula,
              data=data
              .assign(treat = data["after_treatment"]*data["treated"])
              .pipe(demean(col_to_demean="treat"))
              .pipe(demean(col_to_demean="cigsale")))
print(mod.fit().summary().tables[1])

# Synthetic Controls Revisited
# get a synthetic control for the treated unit: The idea here is that Ypost,sc 
# is a good estimator for our missing potential outcome Y(0)post,tr
# If that is the case, the ATT is simply the average of the treated unit in the post-treatment period 
# minus the average of the synthetic control, also in the post treatment period.
sc_model = SyntheticControl()
y_co_pre = data.query("~after_treatment").query("~treated").pivot("year", "state", "cigsale")
y_tr_pre = data.query("~after_treatment").query("treated")["cigsale"]

sc_model.fit(y_co_pre, y_tr_pre)
sc_weights = pd.Series(sc_model.w_, index=y_co_pre.columns, name="sc_w")
sc = data.query("~treated").pivot("year", "state", "cigsale").dot(sc_weights)
att = data.query("treated")["cigsale"][sc.index > 1988].mean() - sc[sc.index > 1988].mean()
print("SC ATT: ", att.round(4))
# This estimate is much smaller than the one we got with Diff-in-Diff

# Synthetic Controls can accommodate non-parallel pre-treatment trends much better, 
# so it is not susceptible to the same bias as Diff-in-Diff. 
# Rather, the process of baking a Synthetic Control enforces parallel trends, 
# at least in the pre-treatment period. As a result, the estimate we get is much smaller and much more plausible.

# We can visualize this estimation process by plotting the realized outcome for California 
# alongside the outcome of the synthetic control. 
# We also plot as dashed lines the post intervention average of both California and the synthetic control. 
# The difference between these lines is the estimated ATT

plt.plot(sc, label="Synthetic Control")
plt.plot(sc.index, data.query("treated")["cigsale"], label="California", color="C1")

calif_avg = data.query("treated")["cigsale"][sc.index > 1988].mean()
sc_avg = sc[sc.index > 1988].mean()

plt.hlines(calif_avg, 1988, 2000, color="C1", ls="dashed")
plt.hlines(sc_avg, 1988, 2000, color="C0", ls="dashed")

plt.title("SC Estimation")
plt.ylabel("Cigarette Sales")
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.legend()
plt.savefig(png_path+"nsc_est.png")
plt.close()

# we can also recast the Synthetic Control estimator as solving the following optimization problem, 
# which is quite similar to the Two-Way Fixed-Effects formulation we used for Diff-in-Diff
# Synthetic Control adds unit weights Wi to the equation. Second, we have time fixed effects Bt
# but no unit fixed effect Ai nor an overall intercept u

# two formulations are actually equivalent, here is the code for it, which yields the exact same ATT estimate:

@curry
def demean_time(df, col_to_demean):
    return df.assign(**{col_to_demean: (df[col_to_demean]
                                        - df.groupby("year")[col_to_demean].transform("mean"))})

data_w_cs_weights = data.set_index("state").join(sc_weights).fillna(1/len(sc_weights))
formula = f"""cigsale ~ -1 + treat"""
mod = smf.wls(formula,
              data=data_w_cs_weights
              .assign(treat = data_w_cs_weights["after_treatment"]*data_w_cs_weights["treated"])
              .pipe(demean_time(col_to_demean="treat"))
              .pipe(demean_time(col_to_demean="cigsale")),
              weights=data_w_cs_weights["sc_w"]+1e-10)
print(mod.fit().summary().tables[1])


# Synthetic Diff-in-Diff
# The first thing we do in this code is to filter out the treated group. Then, we pivot the pre-treated data so that we have the matrix Ypre,co
# Next, we group the post-treatment data to get the average outcome for each control unit in the post-treatment period. 
# We then add a row full of ones to the top of Ypre,co, which will serve as the intercept. 
# Finally, we regress Ypost,co on the pre-treated periods (the rows of Ypre,co) to get the time weights 
# Notice how we add the constraints to have the weights sum up to 1 and be non-negative. 
# Finally, we toss the intercept away and store the time weights in a series.

def fit_time_weights(data, outcome_col, year_col, state_col, treat_col, post_col):
        control = data.query(f"~{treat_col}")
        
        # pivot the data to the (T_pre, N_co) matrix representation
        y_pre = (control
                 .query(f"~{post_col}")
                 .pivot(year_col, state_col, outcome_col))
        
        # group post-treatment time period by units to have a (1, N_co) vector.
        y_post_mean = (control
                       .query(f"{post_col}")
                       .groupby(state_col)
                       [outcome_col]
                       .mean()
                       .values)
        
        # add a (1, N_co) vector of 1 to the top of the matrix, to serve as the intercept.
        X = np.concatenate([np.ones((1, y_pre.shape[1])), y_pre.values], axis=0)
        
        # estimate time weights
        w = cp.Variable(X.shape[0])
        objective = cp.Minimize(cp.sum_squares(w@X - y_post_mean))
        constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        # print("Intercept: ", w.value[0])
        return pd.Series(w.value[1:], # remove intercept
                         name="time_weights",
                         index=y_pre.index)

# Proposition 99
time_weights = fit_time_weights(data,
                                outcome_col="cigsale",
                                year_col="year",
                                state_col="state",
                                treat_col="treated",
                                post_col="after_treatment")
print(time_weights.round(3).tail())

# To understand a bit more about the role of these weights, we can plot LAMBDApre * Ypreco + Lambda0
# as a horizontal line in the pretreatment period that doesn't get zeroed out. 
# Next to it, we plot the average outcome in the post-treatment period. Notice how they align perfectly. 
# We also show the estimated time weights in red bars and in the secondary axis.

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(data.query("~treated").query("~after_treatment").groupby("year")["cigsale"].mean())
ax.plot(data.query("~treated").query("after_treatment").groupby("year")["cigsale"].mean())

intercept = -15.023877689807628
ax.hlines((data.query("~treated").query("~after_treatment").groupby("year")["cigsale"].mean() * time_weights).sum() - 15, 1986, 1988,
          color="C0", ls="dashed", label=""" $\lambda_{pre} Y_{pre, co} + \lambda_0$""")
ax.hlines(data.query("~treated").query("after_treatment").groupby("year")["cigsale"].mean().mean(), 1988, 2000,
          color="C1", ls="dashed", label="""Avg  $Y_{post, co}$""")
ax.vlines(x=1988, ymin=90, ymax=140, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.legend()

plt.title("Time Period Balancing")
plt.ylabel("Cigarette Sales")

ax2 = ax.twinx()
ax2.bar(time_weights.index, time_weights, label="$\lambda$")
ax2.set_ylim(0,10)
ax2.set_ylabel("Time Weights")
plt.savefig(png_path+"time_period_balance.png")
plt.close()

def calculate_regularization(data, outcome_col, year_col, state_col, treat_col, post_col):
    n_treated_post = data.query(post_col).query(treat_col).shape[0]
    first_diff_std = (data
                      .query(f"~{post_col}")
                      .query(f"~{treat_col}")
                      .groupby(state_col)
                      [outcome_col]
                      .diff()
                      .std())
    return n_treated_post**(1/4) * first_diff_std


# As for the unit weights, there is nothing particularly new in them. 
# We can reuse a lot of the code from the function to estimate the time weights. 
# We only need to be careful about the dimensions, since the problem is now upside down.
def fit_unit_weights(data, outcome_col, year_col, state_col, treat_col, post_col):
    zeta = calculate_regularization(data, outcome_col, year_col, state_col, treat_col, post_col)
    pre_data = data.query(f"~{post_col}")
    
    # pivot the data to the (T_pre, N_co) matrix representation
    y_pre_control = (pre_data
                     .query(f"~{treat_col}")
                     .pivot(year_col, state_col, outcome_col))
    
    # group treated units by time periods to have a (T_pre, 1) vector.
    y_pre_treat_mean = (pre_data
                        .query(f"{treat_col}")
                        .groupby(year_col)
                        [outcome_col]
                        .mean())
    
    # add a (T_pre, 1) column to the begining of the (T_pre, N_co) matrix to serve as intercept
    T_pre = y_pre_control.shape[0]
    X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1) 
    
    # estimate unit weights. Notice the L2 penalty using zeta
    w = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.sum_squares(X@w - y_pre_treat_mean.values) + T_pre*zeta**2 * cp.sum_squares(w[1:]))
    constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    
    # print("Intercept:", w.value[0])
    return pd.Series(w.value[1:], # remove intercept
                     name="unit_weights",
                     index=y_pre_control.columns)


unit_weights = fit_unit_weights(data,
                                outcome_col="cigsale",
                                year_col="year",
                                state_col="state",
                                treat_col="treated",
                                post_col="after_treatment")
# first 5 states unit weights
print(unit_weights.round(3).head())


# These unit weights also define a synthetic control that we can plot alongside the outcome of California. '
# We'll also plot the traditional synthetic control we've estimated earlier alongside 
# the one we've just estimated plus the intercept term. 
# Will give us some intuition on what is going on and the difference between what we just did and traditional Synthetic Control

intercept = -24.75035353644767
sc_did = data_piv.drop(columns="california").values @ unit_weights.values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,5))

ax1.plot(data_piv.index, sc_did, label="Synthetic Control (SDID)", color="C0", alpha=.8)
ax1.plot(data_piv["california"], label="California", color="C1")
ax1.vlines(x=1988, ymin=40, ymax=160, linestyle=":", lw=2, label="Proposition 99", color="black")

ax1.legend()
ax1.set_title("SDID Synthetic Control")
ax1.set_ylabel("Cigarette Sales");

ax2.plot(data_piv.index, sc_did+intercept, label="Synthetic Control (SDID) + $w_0$", color="C0", alpha=.8)
ax2.plot(data_piv.index, sc, label="Traditional SC", color="C0", ls="dashed")
ax2.plot(data_piv["california"], label="California", color="C1")
ax2.vlines(x=1988, ymin=40, ymax=160, linestyle=":", lw=2, label="Proposition 99", color="black")
ax2.legend()
ax2.set_title("SDID and Traditional SCs")
plt.savefig(png_path+"sdid.png")
plt.close()

# Now that we have both time LAMBDAt and unit Wt weights
def join_weights(data, unit_w, time_w, year_col, state_col, treat_col, post_col):
    return (
        data
        .set_index([year_col, state_col])
        .join(time_w)
        .join(unit_w)
        .reset_index()
        .fillna({time_w.name: data[post_col].mean(),
                 unit_w.name: data[treat_col].mean()})
        .assign(**{"weights": lambda d: (d[time_w.name]*d[unit_w.name]).round(10)})
        .astype({treat_col:int, post_col:int}))

did_data = join_weights(data, unit_weights, time_weights,
                        year_col="year",
                        state_col="state",
                        treat_col="treated",
                        post_col="after_treatment")
print(did_data.head())

# Finally, all we have to do is estimate a Diff-in-Diff model with the weights we've just defined. 
# The parameter estimate associated with the interaction term for the post-treatment period and 
# treated dummy will be the Synthetic Difference-in-Differences estimate for the ATT
did_model = smf.wls("cigsale ~ after_treatment*treated",
                    data=did_data,
                    weights=did_data["weights"]+1e-10).fit()
print(did_model.summary().tables[1])

# To grasp what SDID is doing, we can plot the Diff-in-Diff lines for the treated (California) and the SDID Synthetic Control
# Notice how we are projecting the trend we see in the synthetic control onto the treated unit to get the counterfactual Y(0)tr,post
# The difference between the two solid purple lines is the estimated ATT
# We start those lines in 1987 to show how the time weights zero out all periods but 1986, 87 and 88. 
# The time weights are also shown in the small plot down below.
avg_pre_period = (time_weights * time_weights.index).sum()
avg_post_period = 1989 + (2000 - 1989) / 2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(data_piv.index, sc_did, label="California")
ax1.plot(data_piv.index, data_piv["california"], label="Synthetic Control (SDID)")
ax1.vlines(1989, data_piv["california"].min(), sc_did.max(), color="black", ls="dotted", label="Prop. 99")

pre_sc = did_model.params["Intercept"]
post_sc = pre_sc + did_model.params["after_treatment"]
pre_treat = pre_sc + did_model.params["treated"]
post_treat = post_sc + did_model.params["treated"] + did_model.params["after_treatment:treated"]

sc_did_y0 = pre_treat + (post_sc - pre_sc)

ax1.plot([avg_pre_period, avg_post_period], [pre_sc, post_sc], color="C2")
ax1.plot([avg_pre_period, avg_post_period], [pre_treat, post_treat], color="C2", ls="dashed")
ax1.plot([avg_pre_period, avg_post_period], [pre_treat, sc_did_y0], color="C2")
ax1.legend()
ax1.set_title("Synthetic Diff-in-Diff")
ax1.set_ylabel("Cigarette Sales")

ax2.bar(time_weights.index, time_weights)
ax2.vlines(1989, 0, 1, color="black", ls="dotted")
ax2.set_ylabel("Time Weights")
ax2.set_xlabel("Years")
plt.savefig(png_path+"synth_diff_in_diff.png")
plt.close()
# The above estimator estimates the ATT which is the effect of Propostion 99 on California averaged out across all post-treatment time periods. 
# But, from the above plot, it looks like the effect increases over time. What if we want to take that into account
# need to deal with effect heterogeneity across time.

# Time Effect Heterogeneity and Staggered Adoption
# All we have to do is run SDID multiple times, one for each time period
# first merge all the steps of SDID into a single function. 
# That is, estimating the unit and time weights and running DiD.

def synthetic_diff_in_diff(data, outcome_col, year_col, state_col, treat_col, post_col):
    
    # find the unit weights
    unit_weights = fit_unit_weights(data,
                                    outcome_col=outcome_col,
                                    year_col=year_col,
                                    state_col=state_col,
                                    treat_col=treat_col,
                                    post_col=post_col)
    
    # find the time weights
    time_weights = fit_time_weights(data,
                                    outcome_col=outcome_col,
                                    year_col=year_col,
                                    state_col=state_col,
                                    treat_col=treat_col,
                                    post_col=post_col)

    # join weights into DiD Data
    did_data = join_weights(data, unit_weights, time_weights,
                            year_col=year_col,
                            state_col=state_col,
                            treat_col=treat_col,
                            post_col=post_col)
    
    # run DiD
    formula = f"{outcome_col} ~ {post_col}*{treat_col}"
    did_model = smf.wls(formula, data=did_data, weights=did_data["weights"]+1e-10).fit()
    
    return did_model.params[f"{post_col}:{treat_col}"]

print(synthetic_diff_in_diff(data, 
                       outcome_col="cigsale",
                       year_col="year",
                       state_col="state",
                       treat_col="treated",
                       post_col="after_treatment"))


# Now that we have a way of easily running SDID, we can run it multiple times, 
# filtering out all the post-treatment periods except the one for which we want the effect.
effects = {year: synthetic_diff_in_diff(data.query(f"~after_treatment|(year=={year})"), 
                                        outcome_col="cigsale",
                                        year_col="year",
                                        state_col="state",
                                        treat_col="treated",
                                        post_col="after_treatment")
           for year in range(1989, 2001)}

effects = pd.Series(effects)
plt.plot(effects)
plt.ylabel("Effect in Cigarette Sales")
plt.title("SDID Effect Estimate by Year")
plt.savefig(png_path+"sdid_eff_est_y.png")
plt.close()

# Conveniently, running multiple SDID will also be important to deal with the staggered adoption case. 
# With staggered addoption design, we have multiple treated units, which get the treatment at different time periods.
np.random.seed(1)
n = 3
tr_state = (data
            .query(f"state.isin({list(np.random.choice(data['state'].unique(), n))})")
            .assign(**{
                "treated": True,
                "state": lambda d: "new_" + d["state"].astype(str),
                "after_treatment": lambda d: d["year"] > 1992
            })
            # effect of 3% / year
            .assign(**{"cigsale": lambda d: np.random.normal(d["cigsale"] - 
                                                             d["cigsale"]*(0.03*(d["year"] - 1992))*d["after_treatment"], 1)}))

new_data = pd.concat([data, tr_state]).assign(**{"after_treatment": lambda d: np.where(d["treated"], d["after_treatment"], np.nan)})
new_data_piv = new_data.pivot("year", "state", "cigsale")

new_tr_states = list(filter(lambda c: str(c).startswith("new"), new_data_piv.columns))

plt.figure(figsize=(10,5))
plt.plot(new_data_piv.drop(columns=["california"]+new_tr_states), color="C1", alpha=0.3)
plt.plot(new_data_piv.drop(columns=["california"]+new_tr_states).mean(axis=1), lw=3, color="C1", ls="dashed", label="Control Avg.")

plt.plot(new_data_piv["california"], color="C0", label="California")
plt.plot(new_data_piv[new_tr_states].mean(axis=1), color="C4", label="New Tr State")

plt.vlines(x=1988, ymin=40, ymax=300, linestyle=":", lw=2, label="Proposition 99", color="black")
plt.vlines(x=1992, ymin=40, ymax=300, linestyle="dashed", lw=2, label="New State Tr.", color="black")
plt.legend()
plt.ylabel("Cigarette Sales")
plt.title("Two Treatment Groups")
plt.savefig(png_path+"two_treat_groups.png")
plt.close()
# ^^^^ 3 new hypothetical states that adopt prop 99 in 1993


# We finally have this staggered adoption data.
#  Now, we need to figure out how to filter out some states so we can break the problem into multiple block assignment cases.
#  First, we can group states by when they passed the law. The following code does exactly that.
# pdb.set_trace()
assignment_blocks = (new_data[(new_data.treated & new_data.after_treatment)]
                     .groupby("state")["year"].min()
                     .reset_index()
                     .groupby("year")["state"].apply(list).to_dict())
print(assignment_blocks)

# As you can see, we have two groups of states. 
# One with only California, which was treated starting in 1989, and another with the three new states we've created, 
# which were all treated starting in 1993. Now, we need to run SDID for each of those groups. 
# We can easily do that, but keeping just the control units plus one of those groups. There is a catch, though. 
# The after_treatment column will have a different meaning, depending on which group we are looking at. 
# If we are looking at the group containing only California, after_treatment should be year >= 1989; 
# if we are looking at the group with the new states, it should be year >= 1993. 
# Fortunately, this is pretty easy to account for. All we need is to recreate the after_treatment in each iteration.
staggered_effects = {year: synthetic_diff_in_diff(new_data
                                                   .query(f"~treated|(state.isin({states}))")
                                                   .assign(**{"after_treatment": lambda d: d["year"] >= year}),
                                                  outcome_col="cigsale",
                                                  year_col="year",
                                                  state_col="state",
                                                  treat_col="treated",
                                                  post_col="after_treatment")
                     for year, states in assignment_blocks.items()}
print(staggered_effects)

# Not surprisingly, the ATT estimate for the first group, the one with only California, 
# is exactly the same as the one we've seen before. The other ATT refers to the one we get with the new group of states. 
# We have to combine them into a single ATT. This can be done with the weighted average we've explained earlier.
# First, we calculate the number of treated entries (after_treatment & treated) in each block. Then, we combine the 
# ATTs using those weights.

weights = {year: sum((new_data["year"] >= year) & (new_data["state"].isin(states)))
           for year, states in assignment_blocks.items()}

att = sum([effect*weights[year]/sum(weights.values()) for year, effect in staggered_effects.items()])

print("weights: ", weights)
print("ATT: ", att)
# Here, we have a total of 36 treatment instances: the usual 12 post-treatment periods for California 
# plus 8 treatment periods (1993-2000) for each of the three new treatment states we've introduced. 
# With that in mind, the weight for the first ATT is 12/36 and for the second ATT, is 24/36, which combines to the result above

# Placebo Variance Estimation
# SDID has better precision (lower error bars) when compared to Synthetic Controls. 
# The reason is that the time and unit fixed effects in SDID capture a ton of the variation in the outcome, 
# which in turn, reduces the variance of the estimator.
# how to place a confidence interval around the SDID estimate

# The idea is to run a series of placebo tests, where we pretend a unit from the control pool is treated, 
# when it actually isn't. Then, we use SDID to estimate the ATT of this placebo test and store its result. 
# We re-run this step multiple times, sampling a control unit each time. 
# In the end, we will have an array of placebo ATTs. The variance of this array is the placebo variance of the SDID effect estimate, 
# which we can use to construct a confidence interval

# function which creates the placebo
def make_random_placebo(data, state_col, treat_col):
    control = data.query(f"~{treat_col}")
    states = control[state_col].unique()
    placebo_state = np.random.choice(states)
    return control.assign(**{treat_col: control[state_col] == placebo_state})
np.random.seed(1)
placebo_data = make_random_placebo(data, state_col="state", treat_col="treated")
print(placebo_data.query("treated").tail())
# In the example above, we've sampled state 39 and we are now pretending that it was treated

# The next thing we need is to compute the SDID estimate with this placebo data and repeat that a bunch of times
from joblib import Parallel, delayed # for parallel processing

def estimate_se(data, outcome_col, year_col, state_col, treat_col, post_col, bootstrap_rounds=400, seed=0, njobs=-1):
    np.random.seed(seed=seed)
    sdid_fn = partial(synthetic_diff_in_diff,
                      outcome_col=outcome_col,
                      year_col=year_col,
                      state_col=state_col,
                      treat_col=treat_col,
                      post_col=post_col)
    
    effects = Parallel(n_jobs=njobs)(delayed(sdid_fn)(make_random_placebo(data, state_col=state_col, treat_col=treat_col))
                                     for _ in range(bootstrap_rounds))
    return np.std(effects, axis=0)

effect = synthetic_diff_in_diff(data,
                                outcome_col="cigsale",
                                year_col="year",
                                state_col="state",
                                treat_col="treated",
                                post_col="after_treatment")

se = estimate_se(data,
                 outcome_col="cigsale",
                 year_col="year",
                 state_col="state",
                 treat_col="treated",
                 post_col="after_treatment")

# The standard deviation can then be used to construct confidence intervals much like we described in the formula above.
print(f"Effect: {effect}")
print(f"Standard Error: {se}")
print(f"90% CI: ({effect-1.65*se}, {effect+1.65*se})")
# Notice that the ATT is not significant in this case, but what's more interesting here 
# is to compare the standard error of the SDID estimate with the one we get from traditional Synthetic Control.

def synthetic_control(data, outcome_col, year_col, state_col, treat_col, post_col):
    x_pre_control = (data
                     .query(f"~{treat_col}")
                     .query(f"~{post_col}")
                     .pivot(year_col, state_col, outcome_col)
                     .values)
    
    y_pre_treat_mean = (data
                        .query(f"~{post_col}")
                        .query(f"{treat_col}")
                        .groupby(year_col)
                        [outcome_col]
                        .mean())
    
    w = cp.Variable(x_pre_control.shape[1])
    objective = cp.Minimize(cp.sum_squares(x_pre_control@w - y_pre_treat_mean.values))
    constraints = [cp.sum(w) == 1, w >= 0]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    
    sc = (data
          .query(f"~{treat_col}")
          .pivot(year_col, state_col, outcome_col)
          .values) @ w.value
    
    y1 = data.query(f"{treat_col}").query(f"{post_col}")[outcome_col]
    att = np.mean(y1 - sc[-len(y1):])
    
    return att

def estimate_se_sc(data, outcome_col, year_col, state_col, treat_col, post_col, bootstrap_rounds=400, seed=0):
    np.random.seed(seed=seed)
    effects = [synthetic_control(make_random_placebo(data, state_col=state_col, treat_col=treat_col), 
                                 outcome_col=outcome_col,
                                 year_col=year_col,
                                 state_col=state_col,
                                 treat_col=treat_col,
                                 post_col=post_col)
              for _ in range(bootstrap_rounds)]
    return np.std(effects, axis=0)

effect_sc = synthetic_control(data,
                              outcome_col="cigsale",
                              year_col="year",
                              state_col="state",
                              treat_col="treated",
                              post_col="after_treatment")

se_sc = estimate_se_sc(data,
                       outcome_col="cigsale",
                       year_col="year",
                       state_col="state",
                       treat_col="treated",
                       post_col="after_treatment")
print(f"Effect: {effect_sc}")
print(f"Standard Error: {se_sc}")
print(f"90% CI: ({effect_sc-1.65*se_sc}, {effect_sc+1.65*se_sc})")

standard_errors = {year: estimate_se(data.query(f"~after_treatment|(year=={year})"), 
                                     outcome_col="cigsale",
                                     year_col="year",
                                     state_col="state",
                                     treat_col="treated",
                                     post_col="after_treatment")
                   for year in range(1989, 2001)}

standard_errors = pd.Series(standard_errors)

plt.figure(figsize=(15,6))

plt.plot(effects, color="C0")
plt.fill_between(effects.index, effects-1.65*standard_errors, effects+1.65*standard_errors, alpha=0.2,  color="C0")

plt.ylabel("Effect in Cigarette Sales")
plt.xlabel("Year")
plt.title("Synthetic DiD 90% Conf. Interval")
plt.xticks(rotation=45)
plt.savefig(png_path+"synth_did_ci.png")
plt.close()

