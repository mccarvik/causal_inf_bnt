"""
Chapter 11 on propensity scores
"""

import pdb
import pydot
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression

import graphviz as gr
from joblib import Parallel, delayed # for parallel processing



png_path = "pngs/ch11/"
style.use("fivethirtyeight")
pd.set_option("display.max_columns", 6)

data = pd.read_csv("./data/learning_mindset.csv")
# print(data.sample(5, random_state=5))

# Students with higher self-reported success expectation are more likely to have joined the growth mindset seminar
# Although the opportunity to participate was random, participation itself is not. We are dealing with a case of non-compliance her
# print(data.groupby("success_expect")["intervention"].mean())

# print(smf.ols("achievement_score ~ intervention", data=data).fit().summary().tables[1])

plt.hist(data["achievement_score"], bins=20, alpha=0.3, label="All")
plt.hist(data.query("intervention==0")["achievement_score"], bins=20, alpha=0.3, color="C2")
plt.hist(data.query("intervention==1")["achievement_score"], bins=20, alpha=0.3, color="C3")
plt.vlines(-0.1538, 0, 300, label="Untreated", color="C2")
plt.vlines(-0.1538+0.4723, 0, 300, label="Treated", color="C3")
plt.legend()
plt.savefig(png_path + "interv_norm.png")
plt.close()

# The propensity score is the conditional probability of receiving the treatment
# sort of function that converts X into the treatment T.
# propensity score makes this middle ground between the variable X and the treatment T
g = gr.Digraph()
g.edge("T", "Y")
g.edge("X", "Y")
g.edge("X", "P(x)")
g.edge("P(x)", "T")
g.render(filename=png_path+"graph")

categ = ["ethnicity", "gender", "school_urbanicity"]
cont = ["school_mindset", "school_achievement", "school_ethnic_minority", "school_poverty", "school_size"]

data_with_categ = pd.concat([
    data.drop(columns=categ), # dataset without the categorical features
    pd.get_dummies(data[categ], columns=categ, drop_first=False)# categorical features converted to dummies
], axis=1)

# print(data_with_categ.shape)


T = 'intervention'
Y = 'achievement_score'
X = data_with_categ.columns.drop(['schoolid', T, Y])

ps_model = LogisticRegression(C=1e6).fit(data_with_categ[X], data_with_categ[T])

data_ps = data.assign(propensity_score=ps_model.predict_proba(data_with_categ[X])[:, 1])

# print(data_ps[["intervention", "achievement_score", "propensity_score"]].head())

weight_t = 1/data_ps.query("intervention==1")["propensity_score"]
weight_nt = 1/(1-data_ps.query("intervention==0")["propensity_score"])
# print("Original Sample Size", data.shape[0])
# print("Treated Population Sample Size", sum(weight_t))
# print("Untreated Population Sample Size", sum(weight_nt))

sns.boxplot(x="success_expect", y="propensity_score", data=data_ps)
plt.title("Confounding Evidence");
plt.savefig(png_path + "ambitious.png")
plt.close()


sns.distplot(data_ps.query("intervention==0")["propensity_score"], kde=False, label="Non Treated")
sns.distplot(data_ps.query("intervention==1")["propensity_score"], kde=False, label="Treated")
plt.title("Positivity Check")
plt.legend()
plt.savefig(png_path + "pos_check.png")
plt.close()

weight = ((data_ps["intervention"]-data_ps["propensity_score"]) /
          (data_ps["propensity_score"]*(1-data_ps["propensity_score"])))

y1 = sum(data_ps.query("intervention==1")["achievement_score"]*weight_t) / len(data)
y0 = sum(data_ps.query("intervention==0")["achievement_score"]*weight_nt) / len(data)

ate = np.mean(weight * data_ps["achievement_score"])

# print("Y1:", y1)
# print("Y0:", y0)
# print("ATE", ate)


# define function that computes the IPTW estimator
def run_ps(df, X, T, y):
    # estimate the propensity score
    ps = LogisticRegression(C=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    
    weight = (df[T]-ps) / (ps*(1-ps)) # define the weights
    return np.mean(weight * df[y]) # compute the ATE

np.random.seed(88)
# run 1000 bootstrap samples
# bootstrap_sample = 80
# ates = Parallel(n_jobs=4)(delayed(run_ps)(data_with_categ.sample(frac=1, replace=True), X, T, Y)
#                           for _ in range(bootstrap_sample))
# ates = np.array(ates)

# print(f"ATE: {ates.mean()}")
# print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")

# sns.distplot(ates, kde=False)
# plt.vlines(np.percentile(ates, 2.5), 0, 30, linestyles="dotted")
# plt.vlines(np.percentile(ates, 97.5), 0, 30, linestyles="dotted", label="95% CI")
# plt.title("ATE Bootstrap Distribution")
# plt.legend();
# plt.savefig(png_path + "ate_bootstrap.png")
# plt.close()

# Propensity score doesn't need to predict the treatment very well. It just needs to include all the confounding variables.
# If we include variables that are very good in predicting the treatment but have no bearing on the outcome this will actually increase the variance of the propensity score estimator

np.random.seed(42)
school_a = pd.DataFrame(dict(T=np.random.binomial(1, .99, 400), school=0, intercept=1))
school_b = pd.DataFrame(dict(T=np.random.binomial(1, .01, 400), school=1, intercept=1))
ex_data = pd.concat([school_a, school_b]).assign(y = lambda d: np.random.normal(1 + 0.1 * d["T"]))
print(ex_data.head())

ate_w_f = np.array([run_ps(ex_data.sample(frac=1, replace=True), ["school"], "T", "y") for _ in range(500)])
ate_wo_f = np.array([run_ps(ex_data.sample(frac=1, replace=True), ["intercept"], "T", "y") for _ in range(500)])

sns.distplot(ate_w_f, kde=False, label="PS W School")
sns.distplot(ate_wo_f, kde=False, label="PS W/O School")
plt.legend();
plt.savefig(png_path + "predict_treatment_too_well.png")
plt.close()

# As you can see, the propensity score estimator that adds the feature school has a humongous variance
# while the one without it is much more well behaved
# Also, since school is not a confounder, the model without it is also not biased
# As I've said, simply predicting the treatment is not what this is about
# We actually need to construct the prediction in a way that controls for confounding, not in a way to predict the treatment.

sns.distplot(np.random.beta(4,1,500), kde=False, label="Non Treated")
sns.distplot(np.random.beta(1,3,500), kde=False, label="Treated")
plt.title("Positivity Check")
plt.legend();
plt.savefig(png_path + "not_much_overlap.png")
plt.close()

# If this happens, it means that positivity is not very strong

# If a treated has a propensity score of, say, 0.9 and the maximum propensity score of the untreated is 0.7,
# we won't have any untreated to compare to the individual with the 0.9 propensity score
# This lack of balancing can generate some bias, because we will have to extrapolate the treatment effect to unknown regions
# Not only that, entities with very high or very low propensity scores have a very high weight, which increases variance
# As a general rule of thumb, you are in trouble if any weight is higher than 20 
# (which happens with an untreated with propensity score of 0.95 or a treated with a propensity score of 0.05).

# Propensity Score Matching
# we can treat the propensity score as an input feature for other models. ex regression:
print(smf.ols("achievement_score ~ intervention + propensity_score", data=data_ps).fit().summary().tables[1])

# We can also use matching on the propensity score
# This time, instead of trying to find matches that are similar in all the X features,
# we can find matches that just have the same propensity score.
# This is a huge improvement on top of the matching estimator, since it deals with the curse of dimensionality
# Also, if a feature is unimportant for the treatment assignment,
# the propensity score model will learn that and give low importance to it when fitting the treatment mechanism.
# Matching on the features, on the other hand, would still try to find matches where individuals are similar on this unimportant feature.

cm = CausalModel(
    Y=data_ps["achievement_score"].values, 
    D=data_ps["intervention"].values, 
    X=data_ps[["propensity_score"]].values
)

cm.est_via_matching(matches=1, bias_adj=True)

print(cm.estimates)
