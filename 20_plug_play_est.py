"""
script for ch 20
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from toolz import curry

png_path = "pngs/ch20/"

# We want to know if there are subgroups of units that respond better or worse to the treatment. 
# That should allow for a much better policy, one where we only treat the ones that will benefit from it.

email = pd.read_csv("./data/invest_email_rnd.csv")
print(email.head())

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

from sklearn.model_selection import train_test_split

np.random.seed(123)
train, test = train_test_split(email, test_size=0.4)
print(train.shape, test.shape)

y = "converted"
T = "em1"
X = ["age", "income", "insurance", "invested"]

ps = train[T].mean()
y_star_train = train[y] * (train[T] - ps)/(ps*(1-ps))

from lightgbm import LGBMRegressor

np.random.seed(123)
cate_learner = LGBMRegressor(max_depth=3, min_child_samples=300, num_leaves=5)
cate_learner.fit(train[X], y_star_train)

test_pred = test.assign(cate=cate_learner.predict(test[X]))
print(test_pred.head())

# Here is a crazy idea: let's transform the outcome variable by multiplying it with the treatment.
# So, if the unit was treated, you would take the outcome and multiply it by 2 (because propensity is 50% = 1 / 0.5)
# If it wasn't treated, you would take the outcome and multiply it by -2. 
# For example, if one of your customers invested BRL 2000,00 and got the email, 
# the transformed target would be 4000. However, if he or she didn't get the email, it would be -4000.
# This seems very odd, because you are saying that the effect of the email can be a negative number, but bare with me. 
# If we do some of math, we can see that, on average or in expectation, this transformed target will be the treatment effect. 
# This is nothing short of amazing. 
# What I'm saying is that by applying this somewhat wacky transformation, I get to estimate something that I can't even observe.

gain_curve_test = cumulative_gain(test_pred, "cate", y="converted", t="em1")
gain_curve_train = cumulative_gain(train.assign(cate=cate_learner.predict(train[X])), "cate", y="converted", t="em1")
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot(gain_curve_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(test, "converted", "em1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.savefig(png_path+"gain_curve.png")
plt.close()

# Another obvious downside of the target transformation method is that it only works for discrete or binary treatments. 
# This is something you see a lot in the causal inference literature.
# Most of the research is done for the binary treatment case, but you don't find a lot about continuous treatments.

prices_rnd = pd.read_csv("./data/ice_cream_sales_rnd.csv")
print(prices_rnd.head())

np.random.seed(123)
train, test = train_test_split(prices_rnd, test_size=0.3)
print(train.shape, test.shape)

# For the continuous case, we don't have that on-off switch. Units are not treated or untreated. 
# Rather, they are all treated, but with different intensities. 
# Therefore, we can't talk about the effect of giving the treatment. 
# Rather, we need to speak in terms of increasing the treatment.
# In other words, we wish to know how the outcome would change if we increase the treatment by some amount (like price)

# This is like estimating the partial derivative of the outcome function on the treatment
# And because we wish to know that for each group (the CATE, not the ATE), we condition on the features
# In plain English, we would transform the original target by subtracting the mean from it, 
# then we would multiply it by the treatment, from which we've also subtracted the mean from
# Finally, we would divide it by the treatment variance. Alas, we have a target transformation for the continuous case.

y_star_cont = (train["price"] - train["price"].mean()
               *train["sales"] - train["sales"].mean())
cate_learner = LGBMRegressor(max_depth=3, min_child_samples=300, num_leaves=5)

np.random.seed(123)
cate_learner.fit(train[["temp", "weekday", "cost"]], y_star_cont)

cate_test_transf_y = cate_learner.predict(test[["temp", "weekday", "cost"]])

test_pred = test.assign(cate=cate_test_transf_y)
print(test_pred.sample(5))

gain_curve_test = cumulative_gain(test.assign(cate=cate_test_transf_y),
                                "cate", y="sales", t="price")

gain_curve_train = cumulative_gain(train.assign(cate=cate_learner.predict(train[["temp", "weekday", "cost"]])),
                                   "cate", y="sales", t="price")


plt.plot(gain_curve_test, label="Test")
plt.plot(gain_curve_train, label="Train")
plt.plot([0, 100], [0, elast(test, "sales", "price")], linestyle="--", color="black", label="Taseline")
plt.legend()
plt.savefig(png_path+"cum_gain.png")
plt.close()

# Having talked about the continuous case, there is still an elephant in the room we need to adress. 
# We've assumed a linearity on the treatment effect. However, that is very rarely a reasonable assumption. 
# Usually, treatment effects saturate in one form or another. 
# it's reasonable to think that demand will go down faster at the first units of price increase, but then it will fall slower
# The problem here is that elasticity or treatment effect changes with the treatment itself
# One solution is to linearize. Use exp and logs. Not always easy, hve to think it thru

