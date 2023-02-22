"""
script for ch28
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


png_path = "pngs/ch28/"


email = pd.read_csv("./data/invest_email.csv")
print(email.head())

print(email.query("em1 == 0").query("em1_ps>0.9"))

confounders = ["age", "income", "insurance", "invested"]
# correlated with treatment and outcome
print(email[confounders + ["em1", "converted"]].corr()[["em1", "converted"]])
# If we fail to account for this confounding bias, our causal estimates will be wrong

# check how the distribution looks like for those that did and didn't get the email em1
plt_df = pd.melt(email[confounders + ["em1"]], ["em1"], confounders)
g = sns.FacetGrid(plt_df, col="variable", hue="em1", col_wrap=4, sharey=False, sharex=False)
for i, ax in enumerate(g.axes):
    iter_df = plt_df.loc[lambda df: df["variable"] == confounders[i]]
    sns.kdeplot(x="value", hue="em1", data=iter_df, ax=ax, fill=True)
    ax.set_xlabel(confounders[i])
plt.savefig(png_path+"no_email_1.png")
plt.close()

# one clear case when there is a strong argument for the propensity score over orthogonal
# When you've stored the probabilities of receiving the treatment while conducting your experiment, 
# propensity score debiasing can be done without having to estimate a model. 

# generate the debiasing weights by using the formula we've seen above. 
# Then, we will resample with replacement from this dataset, using the newly created weights. 
# This means a unit with weight 2 will be resampled twice as often as a unit with weight 1.
np.random.seed(123)
em1_rnd = email.assign(
    em1_w = email["em1"]/email["em1_ps"] + (1-email["em1"])/(1-email["em1_ps"])
).sample(10000, replace=True, weights="em1_w")
np.random.seed(5)
print(em1_rnd.sample(5))

# This resampling should make a new dataset that is debiased. 
# It should have oversampled units that looked like the treated (high em1_ps) but did not get the treatment 
# and those that looked like the control (low em1_ps), but got the treatment.
# If we look at correlations between the treatment and the confounders, we can see that they essentially vanished.
print(em1_rnd[confounders + ["em1", "converted"]].corr()[["em1", "converted"]])


# Moreover, if we look at the confounders distributions by treatment assignment, we can see how nicely they align. 
# This is not 100% proof that the debiasing worked, but it's good evidence of it.
plt_df = pd.melt(em1_rnd[confounders + ["em1"]], ["em1"], confounders)
g = sns.FacetGrid(plt_df, col="variable", hue="em1", col_wrap=4, sharey=False, sharex=False)
for i, ax in enumerate(g.axes):
    iter_df = plt_df.loc[lambda df: df["variable"] == confounders[i]]
    sns.kdeplot(x="value", hue="em1", data=iter_df, ax=ax, fill=True)
    ax.set_xlabel(confounders[i])
plt.savefig(png_path+"debiased.png")
plt.close()

# If we don't have the propensity score stored, we will have to estimate them. 
# In this situation, it becomes less clear when you should use propensity score or orthogonalisation for debiasing.
# Since we don't have the propensity score, we will use a machine learning model to estimate it. 
# The propensity score is closely related to the probability of treatment, 
# so this ML model must be calibrated to output a probability. 
# Not only that, we need to do cross prediction to work around any sort of bias we might have due to overfitting.

from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

t = "em1"
folds = 5
np.random.seed(123)
# makes calibrated Random Forest. 
m_t = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=100, min_samples_leaf=40, max_depth=3),
    cv=3
)

# estimate PS with cross prediction. 
ps_score_m1 = cross_val_predict(m_t, email[confounders], email[t],
                                cv=folds, method="predict_proba")[:, 1]
print(email.assign(ps_score_m1_est = ps_score_m1).head())

# Check calibration of the model:
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(email["em1"], ps_score_m1, n_bins=3)
plt.plot(prob_pred, prob_true, label="Calibrated RF")
plt.plot([.1,.8], [.1, .8], color="grey", linestyle="dashed", label="Perfectly Calibrated")
plt.ylabel("Fraction of Positives")
plt.xlabel("Average Prediction")
plt.legend()
plt.savefig(png_path+"calib.png")
plt.close()

# true propensity score
np.random.seed(123)
em1_rnd_est = email.assign(
    em1_w = email["em1"]/ps_score_m1 + (1-email["em1"])/(1-ps_score_m1)
).sample(10000, replace=True, weights="em1_w")

# still decent amount of correlation
print(em1_rnd_est[confounders + ["em1"]].corr()["em1"])
plt_df = pd.melt(em1_rnd_est[confounders + ["em1"]], ["em1"], confounders)
g = sns.FacetGrid(plt_df, col="variable", hue="em1", col_wrap=4, sharey=False, sharex=False)
for i, ax in enumerate(g.axes):
    iter_df = plt_df.loc[lambda df: df["variable"] == confounders[i]]
    sns.kdeplot(x="value", hue="em1", data=iter_df, ax=ax, fill=True)
    ax.set_xlabel(confounders[i])
plt.savefig(png_path+"dist_prop_calib.png")
plt.close()
# distributions dont align as well as before
# It has to do with propensities scores that are too high or too low.

# observe this unit: weight of 37
print(email.loc[[1014]])
# problem of high variance.
# If we plot the number of replications for each unit in the debiased dataset, 
# we see that a bunch of them appear more than 10 times. 
# Those are treated units with low propensity score or untreated units with high propensity score.
plt.figure(figsize=(10,5))
sns.scatterplot(
    data=em1_rnd.assign(count=1).groupby(em1_rnd.index).agg({"count":"count", "em1_ps": "mean", "em1": "mean"}),
    x="em1_ps",
    y="count",
    hue="em1",
    alpha=0.2
)
plt.title("Replications on Debiased Data")
plt.savefig(png_path+"high_samples.png")
plt.close()

# Positivity or Common Support
# Besides high variance, we can also have problems with positivity. 
# Positivity, or common support is a causal inference assumption which states that there must be sufficient overlap between 
# the characteristics of the treated and the control units. 
# Or, in other words, that everyone has a non zero probability of getting the treatment or the control. 
# If this doesn't happen, we won't be able to estimate a causal effect that is valid for the entire population, 
# only for those we have common support.

sns.displot(data=email, x="em1_ps", hue="em1")
plt.title("Positivity Check");
plt.savefig(png_path+"positivity_overlap.png")
plt.close()

# Lets look at the other emails
print(email.head())
# First thing that jumps the eye is that there are units with zero probability, 
# which already indicates a violation to the positivity assumption. Now, let's look at the features distributions by em3
sns.pairplot(email.sample(1000)[confounders + ["em3"]], hue="em3", plot_kws=dict(alpha=0.3));
plt.savefig(png_path+"em3_dists.png")
plt.close()

# It looks like em3 was only sent to customers that are older than 40 years. 
# Thats a huge problem. If the control has younger folks, 
# but the treatment doesn't, there is no way we can estimate the counterfactual 

# Just to go thru the process and see what happens
em3_weight = (email
              # using a different implementation to avoid division by zero
              .assign(em3_w = np.where(email["em3"].astype(bool), 1/email["em3_ps"], 1/(1-email["em3_ps"])))
              .sample(10000, replace=True, weights="em3_w"))
print(em3_weight[confounders + ["em3"]].corr()["em3"])
# Still huge correlation with age as it could not be debiased
sns.displot(data=email, x="em3_ps", hue="em3")
plt.title("Positivity Check")
plt.savefig(png_path+"pos_check.png")
plt.close()
# Notice how poor the overlap is here. Units with propensity score below 0.4 almost never get the treatment. 
# Not to mention that huge peak at zero.

# But in the industry, violations of the positivity assumption might not be so problematic
# To give an example, if you are a lender wishing to estimate the elasticity of loan amount on probability of default, 
# you will probably not give high loans to people with very low credit scores. 
# Sure, this will violate the positivity assumption, 
# but you are not very interested in estimating the loan amount elasticity for risky customers 
# because you are not intending to give them loans anyway
# With that in mind, let's debias email-3. Of course, we will remove the younger population from the sample.
print(em3_weight.query("age>40")[confounders + ["em3"]].corr()["em3"])
# notice how the correlation between the treatment and the confounders goes away
sns.pairplot(em3_weight.query("age>40").sample(1000)[confounders + ["em3"]], hue="em3", plot_kws=dict(alpha=0.3))
plt.savefig(png_path+"dist_check_old.png")
plt.close()
# distributions look way better too
