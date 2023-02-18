"""
Script for ch12
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression

style.use("fivethirtyeight")
pd.set_option("display.max_columns", 6)

png_path = "pngs/ch12/"

# Doubly Robust Estimation is a way of combining propensity score and 
# linear regression in a way you don't have to rely on either of them.
data = pd.read_csv("./data/learning_mindset.csv")
print(data.sample(5, random_state=5))

# it doesn't seem to be the case that this data is free from confounding. 
# One possible reason for this is that the treatment variable is measured by the student's receipt of the seminar. 
# So, although the opportunity to participate was random, participation is not. 
# We are dealing with a case of non-compliance here. 
# One evidence of this is how the student's success expectation is correlated with the participation in the seminar. 
# Students with higher self-reported high expectations are more likely to have joined the growth mindset seminar.
print(data.groupby("success_expect")["intervention"].mean())

# one-hot
categ = ["ethnicity", "gender", "school_urbanicity"]
cont = ["school_mindset", "school_achievement", "school_ethnic_minority", "school_poverty", "school_size"]

data_with_categ = pd.concat([
    data.drop(columns=categ), # dataset without the categorical features
    pd.get_dummies(data[categ], columns=categ, drop_first=False) # categorical features converted to dummies
], axis=1)
print(data_with_categ.shape)


def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

T = 'intervention'
Y = 'achievement_score'
X = data_with_categ.columns.drop(['schoolid', T, Y])
print(doubly_robust(data_with_categ, X, T, Y))
# Doubly robust estimator is saying that we should expect individuals who attended the mindset seminar 
# to be 0.388 standard deviations above their untreated fellows, in terms of achievements

from joblib import Parallel, delayed # for parallel processing

np.random.seed(88)
# run 1000 bootstrap samples
bootstrap_sample = 1000
ates = Parallel(n_jobs=4)(delayed(doubly_robust)(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                          for _ in range(bootstrap_sample))
ates = np.array(ates)
print(f"ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))
sns.distplot(ates, kde=False)
plt.vlines(np.percentile(ates, 2.5), 0, 20, linestyles="dotted")
plt.vlines(np.percentile(ates, 97.5), 0, 20, linestyles="dotted", label="95% CI")
plt.title("ATE Bootstrap Distribution")
plt.legend()
plt.savefig(png_path + "ate_bootstrap.png")
plt.close()

# First, it is called doubly robust because it only requires one of the models, to be correctly specified
# In the following estimator, replaced the logistic regression that estimates the propensity score 
# by a random uniform variable that goes from 0.1 to 0.9 
# Since this is random, there is no way it is a good propensity score model, 
# but we will see that the doubly robust estimator still manages to produce an estimation 
# that is very close to when the propensity score was estimated with logistic regression.

from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust_wrong_ps(df, X, T, Y):
    # wrong PS model
    np.random.seed(654)
    ps = np.random.uniform(0.1, 0.9, df.shape[0])
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

print(doubly_robust_wrong_ps(data_with_categ, X, T, Y))

# we can see that the variance is slightly higher than when the propensity score was estimated with a logistic regression.
np.random.seed(88)
parallel_fn = delayed(doubly_robust_wrong_ps)
wrong_ps = Parallel(n_jobs=4)(parallel_fn(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                              for _ in range(bootstrap_sample))
wrong_ps = np.array(wrong_ps)
print(f"Original ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))
print(f"Wrong PS ATE 95% CI:", (np.percentile(wrong_ps, 2.5), np.percentile(wrong_ps, 97.5)))

# Now, assume that the propensity score is correctly specified. But the regression is not
# This makes the doubly robust estimator reduce to the propensity score weighting estimator 
# Replaced both regression models with a random normal variable. There is no doubt that 
# regression is not correctly specified. Still, we will see that doubly robust estimation still manages to recover the same 
# output of about 0.38 that we've seen before.

from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust_wrong_model(df, X, T, Y):
    np.random.seed(654)
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    
    # wrong mu(x) model
    mu0 = np.random.normal(0, 1, df.shape[0])
    mu1 = np.random.normal(0, 1, df.shape[0])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

print(doubly_robust_wrong_model(data_with_categ, X, T, Y))

np.random.seed(88)
parallel_fn = delayed(doubly_robust_wrong_model)
wrong_mux = Parallel(n_jobs=4)(parallel_fn(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                               for _ in range(bootstrap_sample))
wrong_mux = np.array(wrong_mux)
print(f"Original ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))
print(f"Wrong Mu ATE 95% CI:", (np.percentile(wrong_mux, 2.5), np.percentile(ates, 97.5)))
