"""
script for ch 21
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from toolz import curry


@curry
def elast(data, y, t):
        # line coeficient for the one variable linear regression 
        return (np.sum((data[t] - data[t].mean())*(data[y] - data[y].mean())) /
                np.sum((data[t] - data[t].mean())**2))


def cumulative_gain(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    
    ## add (rows/size) as a normalizer. 
    return np.array([elast(ordered_df.head(rows), y, t) * (rows/size) for rows in n_rows])

png_path = "pngs/ch21/"

# the goal here is to figure out who will respond better to the email. 
# we will use non-random data to train the models and random data to validate them. 
# Dealing with-non random data is a harder task, because the meta learners will need to debias the data AND estimate the CATE

test = pd.read_csv("./data/invest_email_rnd.csv")
train = pd.read_csv("./data/invest_email_biased.csv")
print(train.head())

y = "converted"
T = "em1"
X = ["age", "income", "insurance", "invested"]

# The first learner we will use is the S-Learner. 
# This is the simplest learner we can think of. We will use a single (hence the S) machine learning model to estimate

from lightgbm import LGBMRegressor
np.random.seed(123)
s_learner = LGBMRegressor(max_depth=3, min_child_samples=30)
s_learner.fit(train[X+[T]], train[y])
s_learner_cate_train = (s_learner.predict(train[X].assign(**{T: 1})) -
                        s_learner.predict(train[X].assign(**{T: 0})))

s_learner_cate_test = test.assign(
    cate=(s_learner.predict(test[X].assign(**{T: 1})) - # predict under treatment
          s_learner.predict(test[X].assign(**{T: 0}))) # predict under control
)
gain_curve_test = cumulative_gain(s_learner_cate_test, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=s_learner_cate_train), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("S-Learner")
plt.savefig(png_path+"s_learn.png")
plt.close()
# In practice, I find that the S-learner is a good first bet for any causal problem, mostly due to its simplicity. \
# Not only that, the S-learner can handle both continuous and discrete treatments, 
# while the rest of the learners in this chapter can only deal with discrete treatments.
# The major disadvantage of the S-learner is that it tends to bias the treatment effect towards zero.
# Since S-learner employs what is usually a regularized machine learning model, regularization can restrict the estimated treatment effect. 
# Even worse, if the treatment is very weak relative to the impact other covariates play in explaining the outcome,
# the S-learner can discard the treatment variable completely

# T-Learner
# The T-learner tries to solve the problem of discarding the treatment entirely by forcing the learner to first split on it. 
# Instead of using a single model, we will use one model per treatment variable. 
# In the binary case, there are only two models that we need to estimate (hence the name T)

np.random.seed(123)
m0 = LGBMRegressor(max_depth=2, min_child_samples=60)
m1 = LGBMRegressor(max_depth=2, min_child_samples=60)

m0.fit(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
m1.fit(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])

# estimate the CATE
t_learner_cate_train = m1.predict(train[X]) - m0.predict(train[X])
t_learner_cate_test = test.assign(cate=m1.predict(test[X]) - m0.predict(test[X]))
gain_curve_test = cumulative_gain(t_learner_cate_test, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=t_learner_cate_train), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend();
plt.title("T-Learner")
plt.savefig(png_path+"t_learn.png")
plt.close()

# You have lots of data for the untreated and very few data for the treated, a pretty common case in many applications, 
# as treatment is often expensive. Now suppose you have some non linearity in the outcome Y, the treatment effect is constant. 
# these models to compute the cate take the linearity of M(T) minus the non linearity of M(~T) will result in a 
# nonlinear CATE (blue line minus red line), which is wrong, since the CATE is constant and equal to 1 in this case
# What happens here is that the model for the untreated can pick up the non linearity, but the model for the treated cannot, 
# because we've used regularization to deal with a small sample size

# X-Learner
# X-Learner has two stages and a propensity score model. The first one is identical to the T-learner
# Now, things start to take a turn. For second stage, we input the treatment effect for the control and for the treated using models above
# Then, we fit two more models to predict those effects
# So we have one model that is wrong because we've input the treatment effects wrongly and 
# another model that is correct because we've imputed those values correctly. 
# Now, we need a way to combine the two in a way that gives more weight to the correct model. 
# Here is where the propensity score model comes to play. We can combine the two second stage models with propensity score
# More generally, weighted average using the propensity score will make sure we give more weight to the CATE model 
# that was estimated where the assigned treatment was more likely. In other words, we will favor the model 
# that was trained using more data.


from sklearn.linear_model import LogisticRegression

np.random.seed(123)
# First, we have the first stage, which is exactly the same as the T-Learner.
# first stage models
m0 = LGBMRegressor(max_depth=2, min_child_samples=30)
m1 = LGBMRegressor(max_depth=2, min_child_samples=30)

# propensity score model
g = LogisticRegression(solver="lbfgs", penalty='none') 

m0.fit(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
m1.fit(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])
                       
g.fit(train[X], train[T]);

# Now, we impute the treatment effect and fit the second stage models on them.
d_train = np.where(train[T]==0,
                   m1.predict(train[X]) - train[y],
                   train[y] - m0.predict(train[X]))

# second stage
mx0 = LGBMRegressor(max_depth=2, min_child_samples=30)
mx1 = LGBMRegressor(max_depth=2, min_child_samples=30)

mx0.fit(train.query(f"{T}==0")[X], d_train[train[T]==0])
mx1.fit(train.query(f"{T}==1")[X], d_train[train[T]==1]);

# Finally, we make corrected predictions using the propensity score model.

def ps_predict(df, t): 
    return g.predict_proba(df[X])[:, t]
    
    
x_cate_train = (ps_predict(train,0)*mx0.predict(train[X]) +
                ps_predict(train,1)*mx1.predict(train[X]))

x_cate_test = test.assign(cate=(ps_predict(test,0)*mx0.predict(test[X]) +
                                ps_predict(test,1)*mx1.predict(test[X])))

# Lets see how our X-Learner does in the test. Again, let's plot the cumulative gain curve.
gain_curve_test = cumulative_gain(x_cate_test, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=x_cate_train), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("X-Learner")
plt.savefig(png_path+"x_learn.png")
plt.close()

