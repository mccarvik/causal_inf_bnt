"""
Srcipt for ch22
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import style
style.use("ggplot")
from toolz import curry

png_path = "pngs/ch22/"

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



# Debiased/Orthogonal ML works for both continuous and discrete treatments
test = pd.read_csv("./data/ice_cream_sales_rnd.csv")
train = pd.read_csv("./data/ice_cream_sales.csv")
print(train.head())

np.random.seed(123)
sns.scatterplot(data=train.sample(1000), x="price", y="sales", hue="weekday")
plt.savefig(png_path + "scatter.png")
plt.close()

# ML for Nuisance Parameters
# One way we can try to remove this bias is by using a linear model to estimate the treatment effect of prices on sales 
# while controlling for the confounders. Notice that we are only interested in the treatment parameter 
# because that's our treatment effect. We are going to call the other parameters nuisance parameters 
# because we don't care about them. But, as it turns out, even if we don't care about them, we have to get them right, 
# because if we don't, our treatment effect will be off. That's sort of annoying.
# For instance, if we think about it, the relationship between temp and sales is probably not linear.
# First, as temperature increases, more people will go to the beach and buy ice cream, so sales will increase. 
# But, at some point, it becomes too hot and people decide it is best to stay home. At that point, sales will drop.

# Thinking about how to model nuisance parameters is already boring with just a few covariates. 
# But what if we had tens or hundreds of them? With modern datasets, this is pretty common. 
# So, what can we do about it? The answer lies the coolest Econometric theorem ever derived.

# Frisch-Waugh-Lovell
# You can estimate all the nuisance parameters separately. First, regress the outcome on the features to get outcome residuals. 
# Then, regress the treatment on the features to get treatment residuals. 
# Finally, regress the outcome residuals on the feature residuals.
# This will yield the exact same estimate as if we regress the outcome on the features and treatment at the same time.
# we estimate treatment effect by first estimating the effects of the covariates on the outcome (sales) and treatment (price).
my = smf.ols("sales~temp+C(weekday)+cost", data=train).fit()
mt = smf.ols("price~temp+C(weekday)+cost", data=train).fit()
# Then, with the residuals, we estimate the ATE of price on sales.
print(smf.ols("sales_res~price_res", 
        data=train.assign(sales_res=my.resid, # sales residuals
                          price_res=mt.resid) # price residuals
       ).fit().summary().tables[1])
# We've estimated the ATE to -4, meaning that each unit increase in price will lower sales by 4 units.
# Now, let's estimate the same parameter, but this time, we will include the treatment and the covariates in the same model.
print(smf.ols("sales~price+temp+C(weekday)+cost", data=train).fit().params["price"])
# Exact same number! This shows that estimating the treatment effect all at once or separating in the FWL steps is mathematically the same

# Double/Debiased ML can be seen as Frisch, Waugh and Lovell theorem on steroids. 
# The idea is very simple: use ML models when constructing the outcome and treatment residuals:

# Power you gain with ML is flexibility. ML is so powerful that it can capture complicated functional forms in the nuisance relationships. 
# But that flexibility is also troublesome, because it means we now have to take into account the possibility of overfitting.

# To see the issue, suppose that your model is overfitting. The result is that the residual will be smaller than it should be.
# It also means that is capturing more than only the relationship between X and Y 
# Part of that something more is the relationship between X and Y and if it is capturing some of that, 
# the residual regression will be biased towards zero. In other words:
# It is capturing the causal relationship and not leaving it to the final residual regression.

# Now to see the problem in overfitting, notice that it will explain more of the variance in T than it should. 
# As a result, the treatment residual will have less variance than it should. If there is less variance in the treatment, 
# the variance of the final estimator will be high. It is as if the treatment is the same for almost everyone. 
# And if everyone has the same treatment level, it becomes very difficult to estimate what would happen under different treatment levels

# Those are the problems we have when using ML models, but how can we correct them? 
# The answer lies in what we will call cross prediction and out-of-fold residuals.
# We will split out data into K parts of equal size. 
# Then, for each part k, we will estimate the ML models on all the other K-1 samples and make the residuals on the k part. 
# Notice that these residuals are made using out-of-fold prediction. 
# We fit the model on one part of the data, but make the predictions and residuals on another part.
# so even if the model does overfit, it won't drive the residuals to zero artificially. 
# Finally, we combine the predictions on all the K parts to estimate the final causal model

# Step by step implementation of the Double/Debiased ML
# First, let's estimate the nuisance relationship using ML models
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict

y = "sales"
T = "price"
X = ["temp", "weekday", "cost"]

debias_m = LGBMRegressor(max_depth=3)

train_pred = train.assign(price_res =  train[T] -
                          cross_val_predict(debias_m, train[X], train[T], cv=5)
                          + train[T].mean()) # add mu_t for visualization. 

# This is the debias model Mt. That's because the role this model is playing on the Double/Debias ML is one of debiasing the treatment.
# The residuals can be viewed as a version of the treatment where all the confounding bias has been removed by the model. 
# In other words, T is orthogonal to X
# Intuitively, T can no longer be explained by X, because it already was.

# To see that, we can show the same plot we've seen earlier but now replacing price with the price residuals. 
# Now, that bias is gone. All the weekdays have the same price residual distribution.
np.random.seed(123)
sns.scatterplot(data=train_pred.sample(1000), x="price_res", y="sales", hue="weekday")
plt.savefig(png_path+"debias1.png")
plt.close()

# The denoising model, My
# Intuitively, is creating a version of the outcome where all the variance due to X has been explained away
# As a result, it becomes easier to do causal estimation. Since it has less noise, causal relationship becomes easier to see

denoise_m = LGBMRegressor(max_depth=3)
train_pred = train_pred.assign(sales_res =  train[y] -
                               cross_val_predict(denoise_m, train[X], train[y], cv=5)
                               + train[y].mean())
# plot same graph as before, now replacing sales with sales residual, we can see that tvariance is much smaller than it was before
np.random.seed(123)
sns.scatterplot(data=train_pred.sample(1000), x="price_res", y="sales_res", hue="weekday");
plt.savefig(png_path+"debias2.png")
plt.close()

# It is now easy to see the negative relationship between prices and sales
# Finally, to estimate that causal relationship, we can run a regression on the residuals.
final_model = smf.ols(formula='sales_res ~ price_res', data=train_pred).fit()
print(final_model.summary().tables[1])

# when we use the residualized or orthogonalised version of sales and price, 
# we can be very confident that the relationship between prices and sales is negative, which makes a lot of sense. 
# As we increase prices, demand for ice cream should fall.

# But if we look at the un-residualized or raw relationship between prices and sales, 
# because of bias, we find a positive relationship. That is because, in anticipation to high sales, prices are increased.
final_model = smf.ols(formula='sales ~ price', data=train_pred).fit()
print(final_model.summary().tables[1])

# Estimating CATE
final_model_cate = smf.ols(formula='sales_res ~ price_res * (temp + C(weekday) + cost)', data=train_pred).fit()
cate_test = test.assign(cate=final_model_cate.predict(test.assign(price_res=1))
                        - final_model_cate.predict(test.assign(price_res=0)))
gain_curve_test = cumulative_gain(cate_test, "cate", y=y, t=T)
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot([0, 100], [0, elast(test, y, T)], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("R-Learner")
plt.savefig(png_path+"elast_curve.png")
plt.close()

# Non Parametric Double/Debiased ML
# can delegate finding the form of the parameter relationships (nonlinear) that to a ML model. 
# In other words, let the machine learn that complicated function form. 
# As it turns out, that's totally possible if we make a few changes to our original Double/Debiased ML algorithm.

y = "sales"
T = "price"
X = ["temp", "weekday", "cost"]

debias_m = LGBMRegressor(max_depth=3)
denoise_m = LGBMRegressor(max_depth=3)
train_pred = train.assign(price_res =  train[T] - cross_val_predict(debias_m, train[X], train[T], cv=5),
                          sales_res =  train[y] - cross_val_predict(denoise_m, train[X], train[y], cv=5))

# Recall that Double/Debiased-ML models the data as follows: where Mt and My are models that, respectively, 
# predicts the outcome and treatment from the features. If we rearrange the terms above, we can isolate the error term
# This is nothing short of awesome, because now we can call this a causal loss function. Which means that, 
# if we minimize the square of this loss, we will be estimating expected value of e which is the CATE.

model_final = LGBMRegressor(max_depth=3)
# create the weights
w = train_pred["price_res"] ** 2 
# create the transformed target
y_star = (train_pred["sales_res"] / train_pred["price_res"])
# use a weighted regression ML model to predict the target with the weights.
model_final.fit(X=train[X], y=y_star, sample_weight=w)
cate_test_non_param = test.assign(cate=model_final.predict(test[X]))
gain_curve_test_non_param = cumulative_gain(cate_test_non_param, "cate", y=y, t=T)
plt.plot(gain_curve_test_non_param, color="C0", label="Non-Parametric")
plt.plot(gain_curve_test, color="C1", label="Parametric")
plt.plot([0, 100], [0, elast(test, y, T)], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("R-Learner")
plt.savefig(png_path+"r_learrn.png")
plt.close()

# non-parametric ML capture this saturating behavior in the treatment effect - nonparametric
np.random.seed(321)
n=5000
discount = np.random.gamma(2,10, n).reshape(-1,1)
discount.sort(axis=0) # for better ploting
sales = np.random.normal(20+10*np.sqrt(discount), 1)
plt.plot(discount, 20 + 10*np.sqrt(discount))
plt.ylabel("Sales")
plt.xlabel("Discount")
plt.savefig(png_path+"nonparametric.png")
plt.close()

# Now, let's apply the Non-Parametric Double/Debias ML to this data.
debias_m = LGBMRegressor(max_depth=3)
denoise_m = LGBMRegressor(max_depth=3)
# orthogonalising step
discount_res =  discount.ravel() - cross_val_predict(debias_m, np.ones(discount.shape), discount.ravel(), cv=5)
sales_res =  sales.ravel() - cross_val_predict(denoise_m, np.ones(sales.shape), sales.ravel(), cv=5)
# final, non parametric causal model
non_param = LGBMRegressor(max_depth=3)
w = discount_res ** 2 
y_star = sales_res / discount_res
non_param.fit(X=discount_res.reshape(-1,1), y=y_star.ravel(), sample_weight=w.ravel());
# With the above model, we can get the CATE estimate. The issue here is that the CATE is not linear. 
# As the treatment increases, the CATE should decrease. 
# The question we are trying to answer is if the non-parametric model can capture that non linearity.

# So, does this mean that the non-parametric model can't capture the non-linearity of the treatment effect? 
# Again, not really... Rather, what is happening is that Double/ML finds the locally linear approximation to the non-linear CATE. 
# it finds the derivative of the outcome with respect to the treatment at that treatment level or around the treatment. 
# This is equivalent to finding the slopes of the lines that are tangential to the outcome function at the treatment point.
# Non-Parametric Double-ML will figure out that the treatment effect will be smaller as we increase the treatment. 
# But, no, it won't find the non-linear treatment effect, but rather the local linear treatment effect. 
# plot those linear approximations against the ground true non-linear causal effect and indeed, they are good approximations.

cate = non_param.predict(X=discount)
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(discount, sales)
plt.plot(discount, 20 + 10*np.sqrt(discount), label="Ground Truth", c="C1")
plt.title("Sales by Discount")
plt.xlabel("Discount")
plt.legend()

plt.subplot(1,2,2)
plt.scatter(discount, cate, label="$\hat{\\tau}(x)$", c="C4")
plt.plot(discount, 5/np.sqrt(discount), label="Ground Truth", c="C2")
plt.title("CATE ($\partial$Sales) by Discount")
plt.xlabel("Discount")
plt.legend()
plt.savefig(png_path+"nonlinear2.png")
plt.close()

# Non-Scientific Double/Debiased ML
# The final idea we will try is a fundamental shift in mentality. 
# We will no longer try to estimate the linear approximation to the CATE. Instead, we will make counterfactual predictions.

# The CATE is the slope of the outcome function at the data point. 
# It is how much we expect the outcome to change if we increase the treatment by a very small amount. 
# More technically, it's the derivative at the point.
# Counterfactual predictions, on the other hand, are an attempt to recreate the entire outcome curve from a single datapoint. 
# Will predict what the outcome would be if treatment were at some other level than the one it currently has, hence the counterfactual.

# If we manage to do so, we will be able to simulate different treatments for a unit and
#  predict how it would respond under those different treatment levels. 
# This is very risky business, because we will be extrapolating an entire curve from a single point.

# once we have this model, we will make 2 step counterfactual predictions. 
# First we will have to make a prediction for the treatment in order to get T, 
# then, we will feed that prediction, along with the features, in our final model


from sklearn.model_selection import KFold

def cv_estimate(train_data, n_splits, model, model_params, X, y):
    cv = KFold(n_splits=n_splits)
    m = model(**model_params)
    
    models = []
    cv_pred = pd.Series(np.nan, index=train_data.index)
    for train, test in cv.split(train_data):
        m.fit(train_data[X].iloc[train], train_data[y].iloc[train])
        cv_pred.iloc[test] = m.predict(train_data[X].iloc[test])
        models += [m]
    
    return cv_pred, models

# Now that we have our own cross prediction function that also gives us the models, 
# we can proceed with the orthogonalisation step.
y = "sales"
T = "price"
X = ["temp", "weekday", "cost"]

debias_m = LGBMRegressor(max_depth=3)
denoise_m = LGBMRegressor(max_depth=3)

y_hat, models_y = cv_estimate(train, 5, LGBMRegressor, dict(max_depth=3), X, y)
t_hat, models_t = cv_estimate(train, 5, LGBMRegressor, dict(max_depth=3), X, T)

y_res = train[y] - y_hat
t_res = train[T] - t_hat

# -1 on price saying that the predictions should not increase as price increases
monotone_constraints = [-1 if col == T else 0 for col in X+[T]]
model_final = LGBMRegressor(max_depth=3, monotone_constraints=monotone_constraints)
model_final = model_final.fit(X=train[X].assign(**{T: t_res}), y=y_res)

# there isn't a clear way to extract the treatment effect from this function. 
# So, rather than extracting a treatment effect, we will input the counterfactual predictions
# We will simulate different price levels for each unit and use our Double-ML model to predict what would be the sales we would see under those different price levels.
# Basically getting points on the sales / price prediction line and extrapolating a function out of those points
# So we can then derive a slope

pred_test = (test
             .rename(columns={"price":"factual_price"})
             .assign(jk = 1)
             .reset_index() # create day ID
             .merge(pd.DataFrame(dict(jk=1, price=np.linspace(3, 10, 9))), on="jk")
             .drop(columns=["jk"]))
print(pred_test.query("index==0"))

def ensamble_pred(df, models, X):
    return np.mean([m.predict(df[X]) for m in models], axis=0)

t_res_test = pred_test[T] - ensamble_pred(pred_test, models_t, X)
pred_test[f"{y}_pred"] = model_final.predict(X=pred_test[X].assign(**{T: t_res_test}))
print(pred_test.query("index==0"))

# we now have a sales prediction for every simulated price. The lower the price, the higher the sales. 
# One interesting thing is that these predictions are off in their level. 
# For instance, they go from about 24 to about -24. 
# That's because the model is predicting the residualized outcome, which is roughly mean zero. 
# This is fine if all you want is to get the slope of the sales curve, which is the price treatment effect
# if you want to fix the prediction levels, all you have to do is add the predictions from the denoising model
y_hat_test = ensamble_pred(pred_test, models_y, X)
pred_test[f"{y}_pred"] = (y_hat_test + 
                          model_final.predict(X=pred_test[X].assign(**{T: t_res_test})))
print(pred_test.query("index==0"))

np.random.seed(1)
sample_ids = np.random.choice(pred_test["index"].unique(), 10)
sns.lineplot(data=pred_test.query("index in @sample_ids"),
             x="price", y="sales_pred", hue="index")
plt.savefig(png_path+"unit_sales_curve.png")
plt.close()

# starting the curves at the same point to show the varying elasticity
np.random.seed(1)
sample_ids = np.random.choice(pred_test["index"].unique(), 10)
sns.lineplot(data=(pred_test
                   .query("index in @sample_ids")
                   .assign(max_sales = lambda d: d.groupby("index")[["sales_pred"]].transform("max"))
                   .assign(sales_pred = lambda d: d["sales_pred"] - d["max_sales"] + d["sales_pred"].mean())),
             x="price", y="sales_pred", hue="index")
plt.savefig(png_path+"same-start.png")
plt.close()



# Potential downsides
# First and foremost, it has the same problems all ML techniques have when applied naively to causal inference: bias. 
# Since the final model is a regularized ML model, this regularization can bias the causal estimate to zero.
# The second problem has to do with the ML algorithm you choose. Here, we choose boosted trees. 
# Trees are not very good at making smooth predictions. As a consequence, we can have discontinuities in the prediction curve. 
# You can see that in the plots above: a stepwise behavior here and there. 
# Also, trees are not very good at extrapolating, so this model might output weird predictions for prices never seen before.

pred_test = (test
             .rename(columns={"price":"factual_price"})
             .assign(jk = 1)
             .reset_index() # create day ID
             .merge(pd.DataFrame(dict(jk=1, price=np.linspace(3, 30, 30))), on="jk")
             .drop(columns=["jk"]))

t_res_test = pred_test[T] - ensamble_pred(pred_test, models_t, X)

y_hat_test = ensamble_pred(pred_test, models_y, X)
pred_test[f"{y}_pred"] = model_final.predict(X=pred_test[X].assign(**{T: t_res_test})) + y_hat_test

np.random.seed(1)
sample_ids = np.random.choice(pred_test["index"].unique(), 10)

sns.lineplot(data=(pred_test
                   .query("index in @sample_ids")
                   .assign(max_sales = lambda d: d.groupby("index")[["sales_pred"]].transform("max"))
                   .assign(sales_pred = lambda d: d["sales_pred"] - d["max_sales"] + d["sales_pred"].mean())),
             x="price", y="sales_pred", hue="index")
plt.savefig(png_path+"tree_issues.png")
plt.close()


