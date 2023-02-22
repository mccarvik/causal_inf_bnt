"""
script for ch24
"""

from toolz import *

import pandas as pd
import numpy as np
from scipy.special import expit

from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

style.use("ggplot")

png_path = "pngs/ch24/"


# Panel data methods are often used in government policy evaluation, 
# but we can easily make an argument about why they are also incredibly useful for the (tech) industry. 
# Companies often track user data across multiple periods of time, which results in a rich panel data structure. 
# Not only that, sometimes experimentation is not possible, so we have to rely on other identification strategies. 
# hypothetical example of a young tech company that tracks the number of people that installed its app across multiple cities.
# At some point in 2021, the tech company launched a new feature in their app. 
# It now wants to know how many new uses that feature brought to the company. The rollout was gradual.
# Diff cities got the feature at different times


date = pd.date_range("2021-05-01", "2021-07-31", freq="D")
cohorts = pd.to_datetime(["2021-06-01", "2021-07-15", "2022-01-01"]).date
units = range(1, 100+1)

np.random.seed(1)
df = pd.DataFrame(dict(
    date = np.tile(date, len(units)),
    unit = np.repeat(units, len(date)),
    cohort = np.repeat(np.random.choice(cohorts, len(units)), len(date)),
    unit_fe = np.repeat(np.random.normal(0, 5, size=len(units)), len(date)),
    time_fe = np.tile(np.random.normal(size=len(date)), len(units)),
    week_day = np.tile(date.weekday, len(units)),
    w_seas = np.tile(abs(5-date.weekday) % 7, len(units)),
)).assign(
    trend = lambda d: (d["date"] - d["date"].min()).dt.days/70,
    day = lambda d: (d["date"] - d["date"].min()).dt.days,
    treat = lambda d: (d["date"] >= d["cohort"]).astype(int),
).assign(
    y0 = lambda d: 10 + d["trend"] + d["unit_fe"] + 0.1*d["time_fe"] + d["w_seas"]/10,
).assign(
    y1 = lambda d: d["y0"] + 1
).assign(
    tau = lambda d: d["y1"] - d["y0"],
    installs = lambda d: np.where(d["treat"] == 1, d["y1"], d["y0"])
)
plt.figure(figsize=(10,4))
[plt.vlines(x=cohort, ymin=9, ymax=15, color=color, ls="dashed") for color, cohort in zip(["C0", "C1"], cohorts[:-1])]
sns.lineplot(
    data=(df
          .groupby(["cohort", "date"])["installs"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs",
    hue="cohort",
)
plt.savefig(png_path+"cohorts.png")
plt.close()

formula = f"""installs ~ treat + C(unit) + C(date)"""
twfe_model = smf.ols(formula, data=df).fit()
print(twfe_model.params["treat"])
print(df.query("treat==1")["tau"].mean())

@curry
def demean(df, col_to_demean):
    return df.assign(**{col_to_demean: (df[col_to_demean]
                                        - df.groupby("unit")[col_to_demean].transform("mean")
                                        - df.groupby("date")[col_to_demean].transform("mean"))})


formula = f"""installs ~ treat"""
mod = smf.ols(formula,
              data=df
              .pipe(demean(col_to_demean="treat"))
              .pipe(demean(col_to_demean="installs")))

result = mod.fit()
print(result.summary().tables[1])

# TWFE = Two Way Fixed Effect
# fixing time and location (or different members on the panel)
# also does the trend projection and level adjustment

# plot the counterfactual predictions
df_pred = df.assign(**{"installs_hat_0": twfe_model.predict(df.assign(**{"treat":0}))})
          

plt.figure(figsize=(10,4))
[plt.vlines(x=cohort, ymin=9, ymax=15, color=color, ls="dashed") for color, cohort in zip(["C0", "C1"], cohorts[:-1])]
sns.lineplot(
    data=(df_pred
          .groupby(["cohort", "date"])["installs_hat_0"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs_hat_0",
    hue="cohort",
    alpha=0.7,
    ls="dotted",
    legend=None
)
sns.lineplot(
    data=(df_pred
          .groupby(["cohort", "date"])["installs"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs",
    hue="cohort",
)
plt.savefig(png_path+"counterfactuals.png")
plt.close()


# Treatment Effect Heterogeneity in Time
# If you ever worked with marketing or tech, you know things take time to mature. 
# If you launch a new feature, it will take time for users to get used to it.
#  Similarly, if you start a marketing campaign, the effect of that campaign won't be instantaneous. 
# It will mature over time and perhaps bring new users even after the campaign is over. 
# This is not the pattern that we had in install data we've seen earlier. 
# There, installs jumped up instantaneously, at the moment the cohort is treated. 
# What happens if we change that to be more in line with what we see in reality. 
# Namely, let's make it so that the ATT is still 1, but now, it takes 10 days to mature 
# (so it will be 0.1 at the first treatment day, 0.2 at the second treatment day and so on, until it reaches 1 on the 10th day). 
# Also, I'll reduce the size of the time and unit effects so that the overall trend is easier to see.


date = pd.date_range("2021-05-01", "2021-07-31", freq="D")
cohorts = pd.to_datetime(["2021-06-01", "2021-07-15", "2022-01-01"]).date
units = range(1, 100+1)

np.random.seed(1)

df_heter = pd.DataFrame(dict(
    date = np.tile(date.date, len(units)),
    unit = np.repeat(units, len(date)),
    cohort = np.repeat(np.random.choice(cohorts, len(units)), len(date)),
    unit_fe = np.repeat(np.random.normal(0, 5, size=len(units)), len(date)),
    time_fe = np.tile(np.random.normal(size=len(date)), len(units)),
    week_day = np.tile(date.weekday, len(units)),
    w_seas = np.tile(abs(5-date.weekday) % 7, len(units)),
)).assign(
    trend = lambda d: (d["date"] - d["date"].min()).dt.days/70,
    day = lambda d: (d["date"] - d["date"].min()).dt.days,
    treat = lambda d: (d["date"] >= d["cohort"]).astype(int),
).assign(
    y0 = lambda d: 10 + d["trend"] + 0.2*d["unit_fe"] + 0.05*d["time_fe"] + d["w_seas"]/50,
).assign(
    y1 = lambda d: d["y0"] + np.minimum(0.1*(np.maximum(0, (d["date"] - d["cohort"]).dt.days)), 1)
).assign(
    tau = lambda d: d["y1"] - d["y0"],
    installs = lambda d: np.where(d["treat"] == 1, d["y1"], d["y0"])
)
plt.figure(figsize=(10,4))
[plt.vlines(x=cohort, ymin=9, ymax=15, color=color, ls="dashed") for color, cohort in zip(["C0", "C1"], cohorts)]
sns.lineplot(
    data=(df_heter
          .groupby(["cohort", "date"])["installs"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs",
    hue="cohort",
)
plt.savefig(png_path+"heterogen.png")
plt.close()


# What we see is that the installs still reach the same level as they did before, but it takes some time (10 days) for that. 
# This seems reasonable right? Most of the data we see in real life works like that, with effects taking some time to mature. 
formula = f"""installs ~ treat + C(date) + C(unit)"""
twfe_model = smf.ols(formula, data=df_heter).fit()
print("Estimated Effect: ", twfe_model.params["treat"])
print("True Effect: ", df_heter.query("treat==1")["tau"].mean())

# First, notice that the true ATT is no longer 1. That is because it will be smaller in the first few periods. 
# Second, and most importantly, we can see is that the estimated ATT from TWFE is not recovering the true ATT anymore. 
# To put it simply:TWFE is biased. But why is that? 
# We have parallel trends, no anticipation and all the other strict exogeneity assumptions here. So what is going on?
# The first step is to realize that TWFE can actually be decomposed into multiple 2 by 2 Diff-in-Diffs. 
# In our example, that would be one that compares: 
# early treated to never treated, 
# late treated against never treated, 
# early treated against late treated (with late treated serving as the control) 
# and late treated against early treated (with early treated being the control)

g_plot_data = (df_heter
               .groupby(["cohort", "date"])["installs"]
               .mean()
               .reset_index()
               .astype({"cohort":str}))
fig, axs = plt.subplots(2, 2, figsize=(15,8), sharex=True, sharey=True)

def plot_comp(df, ax, exclude_cohort, name):
    palette=dict(zip(map(str, cohorts), ["C0", "C1", "C2"]))
    sns.lineplot(
        data=df.query(f"cohort != '{exclude_cohort}'"),
        x="date",
        y="installs",
        hue="cohort",
        palette=palette,
        legend=None,
        ax=ax
    )
    sns.lineplot(
        data=df.query(f"cohort == '{exclude_cohort}'"),
        x="date",
        y = "installs",
        hue="cohort",
        palette=palette,
        alpha=0.2,
        legend=None,
        ax=ax
    )
    ax.set_title(name)

plot_comp(g_plot_data, axs[0,0], cohorts[1], "Early vs Never")
plot_comp(g_plot_data, axs[0,1], cohorts[0], "Late vs Never")
plot_comp(g_plot_data[g_plot_data["date"]<=cohorts[1]], axs[1,0], cohorts[-1], "Early vs Late")
plot_comp(g_plot_data[g_plot_data["date"]>cohorts[0]], axs[1,1], cohorts[-1], "Late vs Early")
plt.tight_layout()
plt.savefig(png_path+"late_v_earyl.png")
plt.close()


# The first three comparisons are no reason for concern, mostly because what they use as control is very well behaved. 
# However, the fourth comparison, late vs early, is problematic. 
# Notice that this comparison uses the early treated as a control. 
# Also notice that this early treated control has a weird behavior. 
# That is a reflection of our ATT not being instantaneous, but instead taking 10 days to mature. 
# Intuitively, we can see that this will mess up the estimation of the counterfactual trend in the DiD, making it steeper than it should be

# Counterfactuals plot
late_vs_early = (df_heter
                 [df_heter["date"].astype(str)>="2021-06-01"]
                 [lambda d: d["cohort"].astype(str)<="2021-08-01"])


formula = f"""installs ~ treat + C(date) + C(unit)"""

twfe_model = smf.ols(formula, data=late_vs_early).fit()

late_vs_early_pred = (late_vs_early
                      .assign(**{"installs_hat_0": twfe_model.predict(late_vs_early.assign(**{"treat":0}))})
                      .groupby(["cohort", "date"])
                      [["installs", "installs_hat_0"]]
                      .mean()
                      .reset_index())


plt.figure(figsize=(10,4))
plt.title("Late vs Early Counterfactuals")
sns.lineplot(
    data=late_vs_early_pred,
    x="date",
    y = "installs",
    hue="cohort",
    legend=None
)

sns.lineplot(
    data=(late_vs_early_pred
          [late_vs_early_pred["cohort"].astype(str) == "2021-07-15"]
          [lambda d: d["date"].astype(str) >= "2021-07-15"]
         ),
    x="date",
    y ="installs_hat_0",
    alpha=0.7,
    color="C0",
    ls="dotted",
    label="counterfactual"
)
plt.savefig(png_path+"counterfact.png")
plt.close()

# New example
date = pd.date_range("2021-05-15", "2021-07-01", freq="D")
cohorts = pd.to_datetime(["2021-06-01", "2021-06-15"])
units = range(1, 100+1)

np.random.seed(1)

df_min = pd.DataFrame(dict(
    date = np.tile(date, len(units)),
    unit = np.repeat(units, len(date)),
    cohort = np.repeat(np.random.choice(cohorts, len(units)), len(date)),
    unit_fe = np.repeat(np.random.normal(0, 5, size=len(units)), len(date)),
)).assign(
    trend = 0,
    day = lambda d: (d["date"] - d["date"].min()).dt.days,
    treat = lambda d: (d["date"] >= d["cohort"]).astype(int),
).assign(
    y0 = lambda d: 10 - d["trend"] + 0.1*d["unit_fe"]
).assign(
    y1 = lambda d: d["y0"] - 0.1*(np.maximum(0, (d["date"] - d["cohort"]).dt.days))
).assign(
    tau = lambda d: d["y1"] - d["y0"],
    installs = lambda d: np.where(d["treat"] == 1, d["y1"], d["y0"])
)
plt.figure(figsize=(10,4))
[plt.vlines(x=cohort, ymin=7, ymax=11, color=color, ls="dashed") for color, cohort in zip(["C0", "C1"], cohorts)]
sns.lineplot(
    data=(df_min
          .groupby(["cohort", "date"])["installs"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs",
    hue="cohort",
)
plt.savefig(png_path+"installs_counter.png")
plt.close()
# Looking at the plot above, we can clearly see that the ATT is negative right? 
# The correct counterfactual should be a straight line at about 11. 
# However, if we run the TWFE estimator, we get a positive effect!
formula = f"""installs ~ treat + C(date) + C(unit)"""
twfe_model = smf.ols(formula, data=df_min).fit()
print(twfe_model.params["treat"])


# Once again focus your attention on the comparison where the early treated cohort serves as the control for the late treated. 
# Remember that, like DiD, TWFE adjusts the trend from the control group to the level of the treated group, 
# so the counterfactual should reflect that.

df_pred = df_min.assign(**{"installs_hat_0": twfe_model.predict(df_min.assign(**{"treat":0}))})
          
plt.figure(figsize=(10,4))
[plt.vlines(x=cohort, ymin=7, ymax=11, color=color, ls="dashed") for color, cohort in zip(["C0", "C1"], cohorts)]
sns.lineplot(
    data=(df_pred
          [(df_pred["cohort"].astype(str) > "2021-06-01") & (df_pred["date"].astype(str) >= "2021-06-15")]
          .groupby(["cohort", "date"])["installs_hat_0"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs_hat_0",
    alpha=0.7,
    ls="dotted",
    color="C0",
    label="counterfactual",
)
sns.lineplot(
    data=(df_pred
          .groupby(["cohort", "date"])["installs"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs",
    hue="cohort",
    legend=None
)
plt.ylabel("Installs");
plt.savefig(png_path+"counters2.png")
plt.close()

# NOTE
# When trying to predict the counterfactual, it looks at the trend of the control group
# but because the treatment as an effect over time, the model thinks that is part of the trend and controls for it
# Making the counterfactuals take on that trend when they shouldnt

# Event Study Design
# I know someone might think we can easily solve this problem by what is called an event study design, 
# where we add one dummy for each period before and after the treatment

df_min_rel = (df_min.assign(relative_days = (df_min["date"] - df_min["cohort"]).dt.days))
print(df_min_rel.head())

# We might think that this formulation would capture the time heterogeneity in the ATT and solve all our issues. 
# Unfortunately, that is not the case. If we try it out / plot the counterfactuals, 
# we see they are far from where they should intuitively be (the horizontal line at 11).

# remove the intercept, otherwise effects will be relative to relative day -30
formula = f"installs ~ -1 + C(relative_days) + C(date) + C(unit)"

twfe_model = smf.ols(formula, data=df_min_rel).fit()
df_pred = df_min_rel.assign(
    installs_hat_0=twfe_model.predict(df_min_rel.assign(relative_days=-1))
) 

plt.figure(figsize=(10,4))
[plt.vlines(x=cohort, ymin=7, ymax=11, color=color, ls="dashed") for color, cohort in zip(["C0", "C1"], cohorts)]
sns.lineplot(
    data=(df_pred
          [(df_pred["cohort"].astype(str) > "2021-06-01") & (df_pred["date"].astype(str) >= "2021-06-15")]
          .groupby(["cohort", "date"])["installs_hat_0"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs_hat_0",
    alpha=0.7,
    ls="dotted",
    color="C0",
    label="counterfactual",
)
sns.lineplot(
    data=(df_pred
          .groupby(["cohort", "date"])["installs"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs",
    hue="cohort",
    legend=None
)
plt.ylabel("Installs")
plt.savefig(png_path+"event_driven_coutners.png")
plt.close()

# we can plot the estimated effects by first extracting the parameter associated with each dummy and 
# then subtracting from them the parameter associated with relative day -1 (the baseline).

effects = (twfe_model.params[twfe_model.params.index.str.contains("relative_days")]
           .reset_index()
           .rename(columns={0:"effect"})
           .assign(relative_day=lambda d: d["index"].str.extract(r'\[(.*)\]').astype(int))
           # set the baseline to period -1
           .assign(effect = lambda d: d["effect"] - d.query("relative_day==-1")["effect"].iloc[0]))

# effects
effects.plot(x="relative_day", y="effect", figsize=(10,4))
plt.ylabel("Estimated Effect")
plt.xlabel("Time Relative to Treatment")
plt.savefig(png_path+"dummy_effect.png")
plt.close()
# The problem here is the same we’ve been discussing. Since we have different timing in the treatment, 
# early treated gets used as a control for late treated units, which causes the model to estimate a very weird counterfactual trend. 
# The bottom line is that adding time relative to treatment dummies does not solve the problem


# A Flexible Functional Form
# the functional form of traditional TWFE is simply not flexible enough to capture this heterogeneity, 
# leading to the sort of bias we've discussed
# A very natural way to group units: by cohort! We know that the effect in an entire cohort follows the same pattern over time. 
# (Reminder: cohort = group of users that has been exposed to the new app and might install)
# So, a natural improvement on that impractical model above is to allow the effect to change by cohort instead of units

formula = f"""installs ~ treat:C(cohort):C(date) + C(unit) + C(date)"""
# for nicer plots latter on
df_heter_str = df_heter.astype({"cohort": str, "date":str})
twfe_model = smf.ols(formula, data=df_heter_str).fit()
# To see if this model works, we can make counterfactual predictions for Y0 by forcing treat to be zero for everyone. 
# Then, we can estimate the effect by taking the observed outcome for the treatment, which is Y1 -Y0
# And see if that matches the true ATT.
df_pred = (df_heter_str
           .assign(**{"installs_hat_0": twfe_model.predict(df_heter_str.assign(**{"treat":0}))})
           .assign(**{"effect_hat": lambda d: d["installs"] - d["installs_hat_0"]}))
print("Number of param.:", len(twfe_model.params))
print("True Effect: ", df_pred.query("treat==1")["tau"].mean())
print("Pred. Effect: ", df_pred.query("treat==1")["effect_hat"].mean())
# it does! We finally managed to make a model which is flexible enough to capture the time heterogeneity, 
# which allowed us to estimate the correct treatment effect! 
# extract the estimated effects by time and cohort and plot them.
# because we know how the data was generated, we know what to expect:
# the effect for each cohort must be zero before treatment, 
# then 1 10 days after treatment and a line climbing up from zero to 1 in the days between the treatment and 10 days after it.
effects = (twfe_model.params[twfe_model.params.index.str.contains("treat")]
           .reset_index()
           .rename(columns={0:"param"})
           .assign(cohort=lambda d: d["index"].str.extract(r'C\(cohort\)\[(.*)\]:'))
           .assign(date=lambda d: d["index"].str.extract(r':C\(date\)\[(.*)\]'))
           .assign(date=lambda d: pd.to_datetime(d["date"]), cohort=lambda d: pd.to_datetime(d["cohort"])))

plt.figure(figsize=(10,4))
sns.lineplot(data=effects, x="date", y="param", hue="cohort")
plt.xticks(rotation=45)
plt.ylabel("Estimated Effect")
plt.savefig(png_path+"est_effect.png")
plt.close()

# feature engiennering to remove unnecessary treatment parameters
def feature_eng(df):
    return (
        df
        .assign(date_0601 = np.where(df["date"]>="2021-06-01", df["date"], "control"),
                date_0715 = np.where(df["date"]>="2021-07-15", df["date"], "control"),)
        .assign(cohort_0601 = (df["cohort"]=="2021-06-01").astype(float),
                cohort_0715 = (df["cohort"]=="2021-07-15").astype(float))
    )

formula = f"""installs ~ treat:cohort_0601:C(date_0601) 
                       + treat:cohort_0715:C(date_0715) 
                       + C(unit) + C(date)"""
twfe_model = smf.ols(formula, data=df_heter_str.pipe(feature_eng)).fit()

df_pred = (df_heter
           .assign(**{"installs_hat_0": twfe_model.predict(df_heter_str
                                                           .pipe(feature_eng)
                                                           .assign(**{"treat":0}))})
           .assign(**{"effect_hat": lambda d: d["installs"] - d["installs_hat_0"]}))


print(len(twfe_model.params))
print("True Effect: ", df_pred.query("treat==1")["tau"].mean())
print("Pred Effect: ", df_pred.query("treat==1")["effect_hat"].mean())
# Same treatment effect, less paramters


# Plotting the treatment effect parameters, 
# we can see how we've removed those from the control cohort and those from the dates before the cohort is treated.

effects = (twfe_model.params[twfe_model.params.index.str.contains("treat")]
           .reset_index()
           .rename(columns={0:"param"})
           .assign(cohort=lambda d: d["index"].str.extract(r':cohort_(.*):'),
                   date_0601=lambda d: d["index"].str.extract(r':C\(date_0601\)\[(.*)\]'),
                   date_0715=lambda d: d["index"].str.extract(r':C\(date_0715\)\[(.*)\]'))
           .assign(date=lambda d: pd.to_datetime(d["date_0601"].combine_first(d["date_0715"]), errors="coerce")))

           
plt.figure(figsize=(10,4))
sns.lineplot(data=effects.dropna(subset=["date"]), x="date", y="param", hue="cohort")
plt.xticks(rotation=45)
plt.savefig(png_path+"rem_params.png")
plt.close()


# plotting counterfactuals for reassurance
twfe_model_wrong = smf.ols("installs ~ treat + C(date) + C(unit)",
                           data=df_pred).fit()


df_pred = (df_pred
           .assign(**{"installs_hat_0_wrong": twfe_model_wrong.predict(df_pred.assign(**{"treat":0}))}))


plt.figure(figsize=(10,4))
sns.lineplot(
    data=(df_pred
          [(df_pred["cohort"].astype(str) > "2021-06-01") & (df_pred["date"].astype(str) >= "2021-06-01")]
          .groupby(["date"])["installs_hat_0"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs_hat_0",
    ls="dotted",
    color="C3",
    label="counterfactual",
)

sns.lineplot(
    data=(df_pred
          .groupby(["cohort", "date"])["installs"]
          .mean()
          .reset_index()),
    x="date",
    y = "installs",
    hue="cohort",
    legend=None
)

plt.ylabel("Installs")
plt.savefig(png_path+"counterfactuals3.png")
plt.close()


