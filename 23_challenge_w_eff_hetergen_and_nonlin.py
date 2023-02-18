"""
Srcipt for ch23
"""

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from matplotlib import style
style.use("ggplot")
from typing import List
from toolz import curry, partial

png_path = "pngs/ch23/"

# We will now see that, sometimes, we can get a better treatment effect segmentation if we don't directly try to estimate CATE, 
# but istead focus on another proxy target, which usually has less variance. 
# A common case when this happens is when the outcome variable of interest is binary.

# Treatment Effects on Binary Outcomes

@curry
def avg_treatment_effect(df, treatment, outcome):
    return df.loc[df[treatment] == 1][outcome].mean() - df.loc[df[treatment] == 0][outcome].mean()
    
    

@curry
def cumulative_effect_curve(df: pd.DataFrame,
                            treatment: str,
                            outcome: str,
                            prediction: str,
                            min_rows: int = 30,
                            steps: int = 100,
                            effect_fn = avg_treatment_effect) -> np.ndarray:
    
    size = df.shape[0]
    ordered_df = df.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    return np.array([effect_fn(ordered_df.head(rows), treatment, outcome) for rows in n_rows])


@curry
def cumulative_gain_curve(df: pd.DataFrame,
                          treatment: str,
                          outcome: str,
                          prediction: str,
                          min_rows: int = 30,
                          steps: int = 100,
                          effect_fn = avg_treatment_effect) -> np.ndarray:
    
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)
    return np.array([effect * (rows / size) for rows, effect in zip(n_rows, cum_effect)])


# Here is an incredibly common problem you might face if you find yourself working for a tech company: 
# management wants to boost customer conversion to your product by means of some sort of nudge. 
# Ex: they want to increase the number of app installs by offering a 10 BRL voucher for customers to make in-app purchases. 
# Since nudges are often expensive, they would love to not have to do it for everyone.
# Rather, it would be great if we could use the conversion boosting nudge only on those customers who are most sensitive to it.
np.random.seed(123)
n = 100000
nudge = np.random.binomial(1, 0.5, n)
age = np.random.gamma(10, 4, n)
estimated_income = np.random.gamma(20, 2, n)*100

latent_outcome = np.random.normal(-4.5 + estimated_income*0.001 + nudge + nudge*age*0.01)
conversion = (latent_outcome > .1).astype(int)

df = pd.DataFrame(dict(conversion=conversion,
                       nudge=nudge,
                       age=age,
                       estimated_income=estimated_income,
                       latent_outcome=latent_outcome))
print(df.mean())
print(avg_treatment_effect(df, "nudge", "latent_outcome"))
print(avg_treatment_effect(df, "nudge", "conversion"))

cumulative_effect_fn = cumulative_effect_curve(df, "nudge", "latent_outcome", min_rows=500)

age_cumm_effect_latent = cumulative_effect_fn(prediction="age")
inc_cumm_effect_latent = cumulative_effect_fn(prediction="estimated_income")

plt.plot(age_cumm_effect_latent, label="age")
plt.plot(inc_cumm_effect_latent, label="est. income")
plt.legend()
plt.xlabel("Percentile")
plt.ylabel("Effect on Latet Outcome")
plt.savefig(png_path+"age_income.png")
plt.close()


# higher the age, the higher the treatment effect
cumulative_effect_fn = cumulative_effect_curve(df, "nudge", "conversion", min_rows=500)

age_cumm_effect_latent = cumulative_effect_fn(prediction="age")
inc_cumm_effect_latent = cumulative_effect_fn(prediction="estimated_income")

plt.plot(age_cumm_effect_latent, label="age")
plt.plot(inc_cumm_effect_latent, label="est. income")
plt.legend()
plt.xlabel("Percentile")
plt.ylabel("Effect on Conversino")
plt.savefig(png_path+"no_latent.png")
plt.close()

# A LOT of treatment effect heterogeneity by estimated_income. 
# Customers with higher estimated_income have much lower treatment effect, 
# which causes the cumulative effect curve to go all the way to zero at the beginning and then slowly converge to the ATE. 
# This tells us that estimated_income will generate segments that have more treatment effect heterogeneity (TEH) 
# compared to the segments we would get with age.

# This is inconvenient right? How come the feature we know to drive effect heterogeneity, age, 
# is worse for personalization when compared with a feature (estimated_income) we know not to modify the treatment effect? 
# The answer lies in the non-linearity of the outcome function.
#  Although estimated_income does not modify the effect of the nudge on the latent outcome,
#  it does once we transform that latent outcome to conversion (at least indirectly).
#  Conversion is not linear. This means that its derivative changes depending on where you are. 
# Since conversion can only go up to 1, if it is already very high, it will be hard to increase it. 
# In other words, the derivative of high conversion is very low.
# But because conversion is also bounded at zero, it will also have a low derivative if it is already very low. 
# Conversion follows an S shape, with low derivatives at both ends. See this by plotting the avg conversion by estimated income bins (bins of 100 by 100).

(df
 .assign(estimated_income_bins=(df["estimated_income"]/100).astype(int)*100)
 .groupby("estimated_income_bins")
 [["conversion"]]
 .mean()
 .plot()
)
plt.savefig(png_path+"S.png")
plt.close()
# Notice how the slope (derivative) of this curve is very small when conversion is very high. 
# It is also small when conversion is very low
# Those very likely to convert or very not likely to convert --> minimal treatment affect

# Experimenting with high and low conversion rates
df["conversion_low"] = conversion = (latent_outcome > 2).astype(int)
df["conversion_high"] = conversion = (latent_outcome > -2).astype(int)
print("Avg. Low Conversion: ", df["conversion_low"].mean())
print("Avg. High Conversion: ", df["conversion_high"].mean())

cumulative_effect_fn = cumulative_effect_curve(df, "nudge", "conversion_low", min_rows=500)
age_cumm_effect_latent = cumulative_effect_fn(prediction="age")
inc_cumm_effect_latent = cumulative_effect_fn(prediction="estimated_income")
plt.plot(age_cumm_effect_latent, label="age")
plt.plot(inc_cumm_effect_latent, label="est. income")
plt.xlabel("Percentile")
plt.ylabel("Effect on Conversino")
plt.legend()
plt.savefig(png_path+"lo_conv.png")
plt.close()

cumulative_effect_fn = cumulative_effect_curve(df, "nudge", "conversion_high", min_rows=500)
age_cumm_effect_latent = cumulative_effect_fn(prediction="age")
inc_cumm_effect_latent = cumulative_effect_fn(prediction="estimated_income")
plt.plot(age_cumm_effect_latent, label="age")
plt.plot(inc_cumm_effect_latent, label="est. income")
plt.xlabel("Percentile")
plt.ylabel("Effect on Conversino")
plt.legend()
plt.savefig(png_path+"hi_conv.png")
plt.close()
# SUMMARY
# when the outcome is binary, the treatment effect tends to be dominated by the curvature (derivative) of the logistic function


# You are working for a streaming company, like Netflix or HBO. 
# A key question the company wants answered is what price to charge customers. 
# In order to answer that, they run an experiment where they randomly assign customers to different priced deals:
# experiment broken down by two customer segments: A, customers with higher estimated income, 
# and B, customers with lower estimated income.

data = pd.DataFrame(dict(
    segment= ["b", "b", "b", "b",  "a", "a", "a", "a",],
    price=[5, 10, 15, 20, ] * 2,
    sales=[5100, 5000, 4500, 3000,  5350, 5300, 5000, 4500]
))

plt.figure(figsize=(8,4))
sns.lineplot(data=data, x="price", y="sales", hue="segment")
plt.title("Avg. Sales by Price (%) by Customer Segment");
plt.savefig(png_path+"netflix.png")
plt.close()

# the ranking of the treatment effect is no longer just between A and B customers. 
# The treatment effect will also depend on where they are in the treatment curve.

plt.figure(figsize=(8,4))
sns.lineplot(data=data, x="price", y="sales", hue="segment")
plt.annotate("1", (8, 5350), bbox=dict(boxstyle="round", fc="1"))
plt.annotate("2", (8, 5000), bbox=dict(boxstyle="round", fc="1"))
plt.annotate("3", (13, 5100), bbox=dict(boxstyle="round", fc="1"))
plt.annotate("4", (13, 4700), bbox=dict(boxstyle="round", fc="1"))
plt.annotate("4", (17, 4800), bbox=dict(boxstyle="round", fc="1"))
plt.annotate("5", (17, 3900), bbox=dict(boxstyle="round", fc="1"))
plt.title("Ordering of the Effect of Increasing Price")
plt.savefig(png_path+"ranked.png")
plt.close()

# Just like in the case where the outcome was binary, in this example, the treatment effect is correlated with the outcome. 
# The higher the sales (lower the price), the lower the absolute treatment effect; the lower the sales (higher the price) the lower the absolute treatment effect. 
# But in this case, the situation is even more complicated because the effect is not only correlated with the outcome, but with the treatment level

data = pd.DataFrame(dict(
    segment= ["b", "b", "b", "b",  "a", "a", "a", "a",],
    price=[5, 10, 15, 20, ] * 2,
    sales=[5100, 5000, 4500, 3000,  5350, 5300, 5000, 4500]
))

plt.figure(figsize=(8,4))
sns.lineplot(data=data.loc[lambda d: (d["segment"] == "a") | (d["price"] < 12) ], x="price", y="sales", hue="segment")
plt.title("Avg. Sales by Price (%) by Customer Segment");
plt.savefig(png_path+"high_seg.png")
plt.close()

# might make sense to linearize and extrapolate from there. Proces has many drawbacks tho
plt.figure(figsize=(8,4))
sns.lineplot(data=data.assign(price = lambda d: -1*(-d["price"]**4)),
             x="price", y="sales", hue="segment")
plt.title("Avg. Sales by -(-price^4)")
plt.savefig(png_path+"linearize.png")
plt.close()

