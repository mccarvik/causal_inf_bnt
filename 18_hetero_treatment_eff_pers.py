"""
script for ch 18
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

png_path = "pngs/ch18/"

# The key thing here is that the decision we want to inform is if we should treat or not.
# Now, we will try to inform another type of decision: who do we treat? 
# we want to estimate the Conditional Average Treatment Effect (CATE)

# Now allow the treatment effect to be different depending on the characteristics of each unit 
# We believe that not all entities respond equally well to the treatment. We want to leverage that heterogeneity. 
# We want to treat only the right units (in the binary case) or figure out what is the optimal treatment dosage for each unit (in the continuous case).

# Goal is to find the elasticity (partial derivative) of each data point given different treatments
# and then apply the treatment to the ones that will benefit the most


prices_rnd = pd.read_csv("./data/ice_cream_sales_rnd.csv")
print(prices_rnd.shape)
print(prices_rnd.head())

np.random.seed(123)
train, test = train_test_split(prices_rnd)

m1 = smf.ols("sales ~ price + temp+C(weekday)+cost", data=train).fit()
print(m1.summary().tables[1])
#  -2.75, in our case. This means that for each additional BRL we charge for our ice cream, 
# we should expect sales to go down by about 3 units.
# Notice how this predicts the exact same elasticity for everyone. 
# Hence, it is not a very good model if we want to know on which days people are less sensitive to ice cream prices
# It estimates the ATE when what we need here is the CATE

m2 = smf.ols("sales ~ price*temp + C(weekday) + cost", data=train).fit()
print(m2.summary().tables[1])
# This second model includes an interaction term between price and temperature. 
# This means that it allows the elasticity to differ for different temperatures. 
# What we are effectively saying here is that people are more or less sensitive to price increases depending on the temperature

# The next model includes interaction terms on all the feature space. 
# This means that elasticity will change with temperature, day of the week and cost.
m3 = smf.ols("sales ~ price*cost + price*C(weekday) + price*temp", data=train).fit()

# first model
def pred_elasticity(m, df, t="price"):
    return df.assign(**{
        "pred_elast": m.predict(df.assign(**{t:df[t]+1})) - m.predict(df)
    })
print(pred_elasticity(m1, test).head())

pred_elast3 = pred_elasticity(m3, test)
np.random.seed(1)
print(pred_elast3.sample(5))

X = ["temp", "weekday", "cost", "price"]
y = "sales"
ml = GradientBoostingRegressor()
ml.fit(train[X], train[y])
# make sure the model is not overfiting.
print(ml.score(test[X], test[y]))

bands_df = pred_elast3.assign(
    elast_band = pd.qcut(pred_elast3["pred_elast"], 2), # create two groups based on elasticity predictions 
    pred_sales = ml.predict(pred_elast3[X]),
    pred_band = pd.qcut(ml.predict(pred_elast3[X]), 2), # create two groups based on sales predictions
)
print(bands_df.head())

# plot a regression line of prices on sales for each partition.
g = sns.FacetGrid(bands_df, col="elast_band")
g.map_dataframe(sns.regplot, x="price", y="sales")
g.set_titles(col_template="Elast. Band {col_name}");
plt.savefig(png_path+"partitions.png")
plt.close()

# Regular ML model, not as stark a differenc
g = sns.FacetGrid(bands_df, col="pred_band")
g.map_dataframe(sns.regplot, x="price", y="sales")
g.set_titles(col_template="Pred. Band {col_name}");
plt.savefig(png_path+"partitions2.png")
plt.close()
# plot conveys a very important point. The predictive model partitions are splitting the units on the y axis
# On days like those in the first partition, we don't sell a lot of ice cream, 
# but we do sell more on days like those in the second partition. 
# the prediction model is doing exactly what it is supposed to do: it predicts sales. 
# It can distinguish between days where there will be low versus high ice cream sales.

# The only problem is that prediction is not particularly useful here. 
# Ultimately, we want to know when we can increase prices and when we can't. 
# But once we look at the slopes of the lines in the predictive model partitions, we see that they don't change much. 
# In other words, both partitions, as defined by the prediction model, have about the same responsiveness to price increase. 
# This doesn't offer us much insight into which are the days we can increase prices, since it looks like price is not affecting sales at all.




