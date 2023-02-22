"""
script for ch30
"""

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
style.use("ggplot")
from toolz import merge
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

png_path = "pngs/ch30/"

def ltv_with_coupons(coupons=None):
    
    n = 10000
    t = 30
    
    np.random.seed(12)

    age = 18 + np.random.poisson(10, n)
    income = 500+np.random.exponential(2000, size=n).astype(int)
    region = np.random.choice(np.random.lognormal(4, size=50), size=n)

    if coupons is None:
        coupons = np.clip(np.random.normal((age-18), 0.01, size=n) // 5 * 5, 0, 15)
    
    assert len(coupons) == n

    np.random.seed(12)
    
    # treatment effect on freq
    freq_mu = 0.5*coupons * age + age
    freq_mu = (freq_mu - 150) / 30
    freq_mu += 2
    

    freq = np.random.lognormal(freq_mu.astype(int))
    churn = np.random.poisson((income-500)/2000 + 22, n)
    ones = np.ones((n, t))
    alive = (np.cumsum(ones, axis=1) <= churn.reshape(n, 1)).astype(int)
    buy = np.random.binomial(1, ((1/(freq+1)).reshape(n, 1) * ones))
    cacq = -1*abs(np.random.normal(region, 2, size=n).astype(int))

    # treatment effect on transactions
    np.random.seed(12)
    transactions = np.random.lognormal(2, size=(n, t)).astype(int) * buy * alive

    transaction_mu = 0.1 + (((income - 500) / 900) * (coupons/8)) + coupons/9
    transaction_mu = np.clip(transaction_mu, 0, 5)
    transaction_mu = np.tile(transaction_mu.reshape(-1,1), t)
    
    np.random.seed(12)
    transactions = np.random.lognormal(transaction_mu, size=(n, t)).astype(int) * buy * alive
    data = pd.DataFrame(merge({"customer_id": range(n), "cacq":cacq},
                              {f"day_{day}": trans 
                               for day, trans in enumerate(transactions.T)}))

    encoded = {value:index for index, value in
           enumerate(np.random.permutation(np.unique(region)))}

    customer_features = pd.DataFrame(dict(customer_id=range(n), 
                                          region=region,
                                          income=income,
                                          coupons=coupons,
                                          age=age)).replace({"region":encoded}).astype(int)
    return data, customer_features


# Causal vs. Predictive problems
transactions, customer_features = ltv_with_coupons()
print(transactions.shape)
print(transactions.head())

print(customer_features.shape)
print(customer_features.head())

def process_data(transactions, customer_data):

    profitable = (transactions[["customer_id"]]
                  .assign(net_value = transactions
                          .drop(columns="customer_id")
                          .sum(axis=1)))

    return (customer_data
            # join net_value and features
            .merge(profitable, on="customer_id")
            # include the coupons cost
            .assign(net_value = lambda d: d["net_value"] - d["coupons"]))

customer_features = process_data(transactions, customer_features)
print(customer_features.head())

print(customer_features.groupby("coupons")["customer_id"].count())
print(customer_features.corr()[["coupons"]])
# older person much more likely to get coupons

# negative correlation between coupons and net_value

sns.barplot(data=customer_features, x="coupons", y="net_value")
plt.title("Net Value by Coupon Value")
plt.savefig(png_path+"cpn_val.png")
plt.close()


simple_policy = 5 * np.ones(customer_features["coupons"].shape)

transactions_simple_policy, customer_features_simple_policy = ltv_with_coupons(simple_policy)
customer_features_simple_policy = process_data(transactions_simple_policy, customer_features_simple_policy)

# Give every one a 5 cpn
print(customer_features_simple_policy.head())

simple_policy_gain = customer_features_simple_policy["net_value"].mean()
print(simple_policy_gain)

# Maybe 5 BRL is the optimal strategy for most of the customers, but not for all of them
# If we can identify the ones where the optimal value is different, we can build a coupon strategy better than the simple one we did above.
# a personalisation problem

def model_bands(train_set, features, target, model_params, n_bands, seed=1):
    
    np.random.seed(seed)
    
    # train the ML model
    reg = ensemble.GradientBoostingRegressor(**model_params)
    reg.fit(train_set[features], train_set[target])
    
    # fit the bands
    bands = pd.qcut(reg.predict(train_set[features]), q=n_bands, retbins=True)[1]
    
    def predict(test_set):
        # make predictions with trained model
        predictions = reg.predict(test_set[features])
        
        # discretize predictions into bands.
        pred_bands = np.digitize(predictions, bands, right=False) 
        return test_set.assign(predictions=predictions,
                               # cliping avoid creating new upper bands
                               pred_bands=np.clip(pred_bands, 1, n_bands))
    return predict

train, test = train_test_split(customer_features, test_size=0.3, random_state=1)
model_params = {'n_estimators': 150,
                'max_depth': 4,
                'min_samples_split': 10,
                'learning_rate': 0.01,
                'loss': 'ls'}

features = ["region", "income", "age"]
target = "net_value"

np.random.seed(1)
model = model_bands(train, features, target, model_params, n_bands=10)
print(model(train).head())

print("Train Score:, ", r2_score(train["net_value"], model(train)["predictions"]))
print("Test Score:, ", r2_score(test["net_value"], model(test)["predictions"]))
# this performance is only the predictive performance. What we really want to know is if this model can make us some money.

# bands by optimal cpn policy
plt.figure(figsize=(12,6))
sns.barplot(data=model(customer_features), x="pred_bands", y="net_value", hue="coupons")
plt.title("Net Value by Coupon Value")
plt.savefig(png_path+"cpn_bands.png")
plt.close()

# First, we will group our customers by band and coupon value and take the average net_value for each group, much like the plot above.
pred_bands = (model(customer_features)
              .groupby(["pred_bands", "coupons"])
              [["net_value"]].mean()
              .reset_index())
print(pred_bands.head(7))


# Then, we will group by band and take the net_value rank for each row. 
# This will order the rows according to the average net_value, where 1 is the best net_value in that band.
pred_bands["max_net"] = (pred_bands
                         .groupby(['pred_bands'])
                         [["net_value"]]
                         .rank(ascending=False))
print(pred_bands.head(7))

# For example, for band one, the best coupon strategy is 10 BRL. 
# Next, we will keep only the greatest net_value per band.
best_coupons_per_band = pred_bands.query("max_net==1")[["pred_bands", "coupons"]]
print(best_coupons_per_band)

coupons_per_id = (model(customer_features)
                 .drop(columns=["coupons"])
                 .merge(best_coupons_per_band, on="pred_bands")
                 [["customer_id", "coupons"]]
                 .sort_values('customer_id'))
print(coupons_per_id.head())

# Finally, to evaluate the policy, we pass the coupons column as the coupon array to the ltv_with_coupons function. 
# This will regenerate the data, now assuming the coupons were given as we defined by this policy.
transactions_policy_w_model, customer_features_policy_w_model = ltv_with_coupons(
    coupons_per_id[["coupons"]].values.flatten()
)

customer_features_policy_w_model = process_data(transactions_policy_w_model, customer_features_policy_w_model)
print(customer_features_policy_w_model.head())

# To check how much money this policy is making us, we can compute the average net_value mean for this new dataset.
policy_w_model_gain = customer_features_policy_w_model["net_value"].mean()
print(policy_w_model_gain)
# We were making more money before this!

plt.figure(figsize=(10,6))
sns.histplot(data=customer_features_policy_w_model, bins=40,
             x="net_value", label="Policy W/ Model", color="C0")
sns.histplot(data=customer_features_simple_policy, bins=40,
             x="net_value", label="Simple Policy", color="C1")
plt.legend()
plt.title(f"Simple Policy Gain: {simple_policy_gain}; Policy w/ Model Gain: {policy_w_model_gain};")
plt.savefig(png_path+"gain_is_less.png")
plt.close()

# From the predictive point of view, this is awesome. It means that your model has captured all the variation in net_income. 
# However, from the policy perspective, this is terrible, because there is no variance left in net_income for us to see 
# how it would change given different coupon values. 
# Without this variation in net_income, it would look like changing the coupon values has no effect on net_income at all, 
# leaving us no room for optimization.


# To summarize it, whenever we want to optimise some Y variable using some T variable, predicting Y will not only not help, it will hurt our policy, 
# since data partitions defined by the prediction will have limited Y  variance, 
# hindering our capacity to estimate how T changes Y, that is, the elasticity 

# The key to fixing this mistake lies in adjusting our objective to what we really want. Instead of estimating Y out of X, 
# which is what prediction does, we need to estimate elasticity out of X

# sometimes, you might actually get away when using a prediction model to achieve a causal inference goal. 
# But for this to happen, Y and elasticity must be somehow correlated
# Ex: waiting in line to get answered by customers services impact customer satisfaction. 
# In this case, we can see that customer satisfaction drops pretty fast in the first few minutes of waiting time. 
# Customers get really pissed off when they go from not having to wait much to having to wait just a little bit. 
# However, as waiting time increases, customer satisfaction is already so low it doesn't drop much afterwards. 
# It sorts of saturates at a lower level.

