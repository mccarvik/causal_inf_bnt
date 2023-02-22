"""
script for ch27
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import statsmodels.formula.api as smf
import statsmodels.api as sm

from nb21 import cumulative_elast_curve_ci, elast, cumulative_gain_ci

png_path = "pngs/ch27/"

# The idea was to estimate the elasticity as the coefficient of a single variable linear regression of y ~ t. 
# However, this only works if the treatment is randomly assigned. 
# If it isn't, we get into trouble due to omitted variable bias.

# To workaround this, we need to make the data look as if the treatment is randomly assigned. 
# I would say there are two main techniques to do this. 
# One is using propensity score and the other using orthogonalization. We will cover the latter in this chapter.
# I would argue that probably the safest way out of non random data is to go out and do some sort of experiment to gather random data. 
# don't trust very much on debiasing techniques because you can never know if you've accounted for every confounder. 
# Having said that, orthogonalization is still very much worth learning. 

# partial derivative: how much would Y increase if I increase one feature while holding all the others fixed.


# what linear regression is doing:
# To get the coefficient of one variable Xk, regression first uses all the other variables to predict Xk and takes the residuals. 
# This "cleans" Xk of any influence from those variables. That way, when we try to understand Xk's impact on Y,
# it will be free from omitted variable bias. Second, regression uses all the other variables to predict Y
# and takes the residuals. This "cleans" Y from any influence from those variables, reducing the variance of Y
# so that it is easier to see how Xk impacts Y

# only take the sample where prices where not randomly assigned. Once again, we separate them into a training and a test set.
# Since we will use the test set to evaluate our causal model, let's see how we can use orthogonalization to debias it.

prices = pd.read_csv("./data/ice_cream_sales.csv")
train, test = train_test_split(prices, test_size=0.5)
print(train.shape, test.shape)

# If we show the correlations on the test set, we can see that price is positively correlated with sales,
# meaning that sales should go up as we increase prices. 
# This is obviously nonsense. People don't buy more if ice cream is expensive. We probably have some sort of bias here.
print(test.corr())

# Weekends (Saturday and Sunday) have higher price but also higher sales
np.random.seed(123)
sns.scatterplot(data=test.sample(1000), x="price", y="sales", hue="weekday")
plt.savefig(png_path+"weekends.png")
plt.close()

# To debias this dataset we will need two models. The first model, let's call it Mt(X), 
# predicts the treatment (price, in our case) using the confounders. 
# It's the one of the stages we've seen above, on the Frisch–Waugh–Lovell theorem.

m_t = smf.ols("price ~ cost + C(weekday) + temp", data=test).fit()
debiased_test = test.assign(**{"price-Mt(X)":test["price"] - m_t.predict(test)})

# Another way of saying this is that the bias has been explained away by the model Mt(x), producing t,
# which is as good as randomly assigned. Of course this only works if we have in X all the confounders that cause both Y and T
# We can also plot this data to see what it looks like.
np.random.seed(123)
sns.scatterplot(data=debiased_test.sample(1000), x="price-Mt(X)", y="sales", hue="weekday")
plt.vlines(0, debiased_test["sales"].min(), debiased_test["sales"].max(), linestyles='--', color="black");
plt.savefig(png_path+"debias.png")
plt.close()

# we can also construct residuals for the outcome.
# the only thing left to explain y-resid is something we didn't used to construct it (not included in X),
# which is only the treatment (again, assuming no unmeasured confounders).

m_y = smf.ols("sales ~ cost + C(weekday) + temp", data=test).fit()
debiased_test = test.assign(**{"price-Mt(X)":test["price"] - m_t.predict(test),
                               "sales-My(X)":test["sales"] - m_y.predict(test)})
np.random.seed(123)
sns.scatterplot(data=debiased_test.sample(1000), x="price-Mt(X)", y="sales-My(X)", hue="weekday")
plt.vlines(0, debiased_test["sales-My(X)"].min(), debiased_test["sales-My(X)"].max(), linestyles='--', color="black");
plt.savefig(png_path+"sales_px_go_down_now.png")
plt.close()

# causal model for price elasticity using the training data.
m3 = smf.ols(f"sales ~ price*cost + price*C(weekday) + price*temp", data=train).fit()

# Then, we'll make elasticity predictions on the debiased test set.
def predict_elast(model, price_df, h=0.01):
    return (model.predict(price_df.assign(price=price_df["price"]+h))
            - model.predict(price_df)) / h

debiased_test_pred = debiased_test.assign(**{
    "m3_pred": predict_elast(m3, debiased_test),
})
print(debiased_test_pred.head())

plt.figure(figsize=(10,6))

cumm_elast = cumulative_elast_curve_ci(debiased_test_pred, "m3_pred", "sales-My(X)", "price-Mt(X)", min_periods=50, steps=200)
x = np.array(range(len(cumm_elast)))
plt.plot(x/x.max(), cumm_elast, color="C0")

plt.hlines(elast(debiased_test_pred, "sales-My(X)", "price-Mt(X)"), 0, 1, linestyles="--", color="black", label="Avg. Elast.")
plt.xlabel("% of Top Elast. Customers")
plt.ylabel("Elasticity of Top %")
plt.title("Cumulative Elasticity")
plt.legend()
plt.savefig(png_path+"cum_elast.png")
plt.close()


plt.figure(figsize=(10,6))
cumm_gain = cumulative_gain_ci(debiased_test_pred, "m3_pred", "sales-My(X)", "price-Mt(X)", min_periods=50, steps=200)
x = np.array(range(len(cumm_gain)))
plt.plot(x/x.max(), cumm_gain, color="C1")

plt.plot([0, 1], [0, elast(debiased_test_pred, "sales-My(X)", "price-Mt(X)")], linestyle="--", label="Random Model", color="black")

plt.xlabel("% of Top Elast. Customers")
plt.ylabel("Cumulative Gain")
plt.title("Cumulative Gain on Debiased Sample")
plt.legend()
plt.savefig(png_path+"cum_gain.png")
plt.close()

# biased data much worse outcome
plt.figure(figsize=(10,6))
cumm_gain = cumulative_gain_ci(debiased_test_pred, "m3_pred", "sales", "price", min_periods=50, steps=200)
x = np.array(range(len(cumm_gain)))
plt.plot(x/x.max(), cumm_gain, color="C1")

plt.plot([0, 1], [0, elast(debiased_test_pred, "sales", "price")], linestyle="--", label="Random Model", color="black")
plt.xlabel("% of Top Elast. Customers")
plt.title("Cumulative Gains on Biased Sample")
plt.ylabel("Cumulative Gains")
plt.legend()
plt.savefig(png_path+"biased_cum_gain.png")
plt.close()

# Orthogonalization with Machine Learning
# There is a catch, though. As we know very well, machine learning models are so powerful that they can fit the data perfectly, 
# or rather, overfit. Just by looking at the equations above, we can know what will happen in that case. 
# If model somehow overfits, the residuals will all be very close to zero. If that happens, it will be hard to find how treatment affects it. 
# Similarly, if model somehow overfits, its residuals will also be close to zero. 
# Hence, there won't be variation in the treatment residual to see how it can impact the outcome.

# To account for that, we need to do sample splitting. 
# That is, we estimate the model with one part of the dataset and we make predictions in the other part. 
# The simplest way to do this is to split the test sample in half, 
# make two models in such a way that each one is estimated in one half of the dataset and makes predictions in the other half.
# A slightly more elegant implementation uses K-fold cross validation. 
# The advantage being that we can train all the models on a sample which is bigger than half the test set.
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

X = ["cost", "weekday", "temp"]
t = "price"
y = "sales"

folds = 5

np.random.seed(123)
m_t = RandomForestRegressor(n_estimators=100)
t_res = test[t] - cross_val_predict(m_t, test[X], test[t], cv=folds)

m_y = RandomForestRegressor(n_estimators=100)
y_res = test[y] - cross_val_predict(m_y, test[X], test[y], cv=folds)
# Now that we have the residuals, let's store them as columns on a new dataset.

ml_debiased_test = test.assign(**{
    "sales-ML_y(X)": y_res,
    "price-ML_t(X)": t_res,
})
print(ml_debiased_test.head())

np.random.seed(123)
sns.scatterplot(data=ml_debiased_test.sample(1000),
                x="price-ML_t(X)", y="sales-ML_y(X)", hue="weekday");
plt.savefig(png_path+"debias_ml.png")
plt.close()
