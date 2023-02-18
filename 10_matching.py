"""
Script for ch10
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import graphviz as gr
import os

style.use("fivethirtyeight")

png_path = "pngs/ch10/"

# regression partitions the data into the dummy cells and computes the mean difference between test and control. 
# This difference in means keeps the Xs constant, since we are doing it in a fixed cell of X dummy
# The way it does this is by applying weights to the cell proportional to the variance of the treatment on that group.

# let's suppose I'm trying to estimate the effect of a drug and I have 6 men and 4 women. 
# My response variable is days hospitalised and I hope my drug can lower that. 
# On men, the true causal effect is -3, so the drug lowers the stay period by 3 days. On women, it is -2. 
# Men are much more affected by this illness and stay longer at the hospital. They also get much more of the drug. 
# Only 1 out of the 6 men does not get the drug. On the other hand, women are more resistant to this illness, so they stay less at the hospital. 
# 50% of the women get the drug.
drug_example = pd.DataFrame(dict(
    sex= ["M","M","M","M","M","M", "W","W","W","W"],
    drug=[1,1,1,1,1,0,  1,0,1,0],
    days=[5,5,5,5,5,8,  2,4,2,4]
))

# Note that simple comparison of treatment and control yields a negatively biased effect, that is, the drug seems less effective than it truly is. 
# This is expected, since we've omitted the sex confounder. 
# In this case, the estimated ATE is smaller than the true one because men get more of the drug and are more affected by the illness.

print(drug_example.query("drug==1")["days"].mean() - drug_example.query("drug==0")["days"].mean())

# using regression, controlling for sex
print(smf.ols('days ~ drug + C(sex)', data=drug_example).fit().summary().tables[1])
# regression uses weights that are proportional to the variance of the treatment in that group. 
# In our case, the variance of the treatment in men is smaller than in women, since only one man is in the control group. 
# So regression will give a higher weight to women in our example and the ATE will be a bit closer to the women's ATE of -2.

# Matching
# Since some sort of confounder X makes it so that treated and untreated are not initially comparable, 
# I can make them so by matching each treated unit with a similar untreated unit. 
# It is like I'm finding an untreated twin for every treated unit.
# By making such comparisons, treated and untreated become again comparable.

trainee = pd.read_csv("./data/trainees.csv")
print(trainee.query("trainees==1"))
print(trainee.query("trainees==0"))
print(trainee.query("trainees==1")["earnings"].mean() - trainee.query("trainees==0")["earnings"].mean())
# notice that trainees are much younger than non trainees, which indicates that age is probably a confounder

# make dataset where no one has the same age
unique_on_age = (trainee
                 .query("trainees==0")
                 .drop_duplicates("age"))
matches = (trainee
           .query("trainees==1")
           .merge(unique_on_age, on="age", how="left", suffixes=("_t_1", "_t_0"))
           .assign(t1_minuts_t0 = lambda d: d["earnings_t_1"] - d["earnings_t_0"]))
print(matches.head(7))
# last column has the difference in earnings between the treated and its matched untreated unit. 
# If we take the mean of this last column we get the ATET estimate while controlling for age
# ATET = Average Treatment Effect of Treated
print(matches["t1_minuts_t0"].mean())


# euclidian distance to find the level of matching
med = pd.read_csv("data/medicine_impact_recovery.csv")
print(med.head())

# no conditioning, medicine causing more harm to the patient
print(med.query("medication==1")["recovery"].mean() - med.query("medication==0")["recovery"].mean())

# control for all X
# scale features
X = ["severity", "age", "sex"]
y = "recovery"
med = med.assign(**{f: (med[f] - med[f].mean())/med[f].std() for f in X})
print(med.head())

# Now, to the matching itself. Instead of coding a matching function, we will use the K nearest neighbour algorithm from Sklearn.
# This algorithm makes predictions by finding the nearest data point in an estimation or training set.
# For matching, we will need 2 of those. One, mt0, will store the untreated points and will find matches in the untreated when asked to do so. 
# The other, mt1, will store the treated point and will find matches in the treated when asked to do so
# After this fitting step, we can use these KNN models to make predictions, which will be our matches.
from sklearn.neighbors import KNeighborsRegressor

treated = med.query("medication==1")
untreated = med.query("medication==0")

mt0 = KNeighborsRegressor(n_neighbors=1).fit(untreated[X], untreated[y])
mt1 = KNeighborsRegressor(n_neighbors=1).fit(treated[X], treated[y])

predicted = pd.concat([
    # find matches for the treated looking at the untreated knn model
    treated.assign(match=mt0.predict(treated[X])),
    
    # find matches for the untreated looking at the treated knn model
    untreated.assign(match=mt1.predict(untreated[X]))
])
print(predicted.head())
print(np.mean((2*predicted["medication"] - 1)*(predicted["recovery"] - predicted["match"])))
# negative now

# Matching Bias
# It turns out the matching estimator as we've designed above is biased. 
# Bias arises when the matching discrepancies are huge
# Fortunately, we know how to correct it. Each observation contributes to the bias.
# So all we need to do is subtract this quantity from each matching comparison in our estimator

from sklearn.linear_model import LinearRegression

# fit the linear regression model to estimate mu_0(x)
ols0 = LinearRegression().fit(untreated[X], untreated[y])
ols1 = LinearRegression().fit(treated[X], treated[y])

# find the units that match to the treated
treated_match_index = mt0.kneighbors(treated[X], n_neighbors=1)[1].ravel()

# find the units that match to the untreatd
untreated_match_index = mt1.kneighbors(untreated[X], n_neighbors=1)[1].ravel()
predicted = pd.concat([
    (treated
     # find the Y match on the other group
     .assign(match=mt0.predict(treated[X])) 
     
     # build the bias correction term
     .assign(bias_correct=ols0.predict(treated[X]) - ols0.predict(untreated.iloc[treated_match_index][X]))),
    (untreated
     .assign(match=mt1.predict(untreated[X]))
     .assign(bias_correct=ols1.predict(untreated[X]) - ols1.predict(treated.iloc[untreated_match_index][X])))
])
print(predicted.head())
# with bias correction
print(np.mean((2*predicted["medication"] - 1)*((predicted["recovery"] - predicted["match"])-predicted["bias_correct"])))

from causalinference import CausalModel

cm = CausalModel(
    Y=med["recovery"].values, 
    D=med["medication"].values, 
    X=med[["severity", "age", "sex"]].values
)

cm.est_via_matching(matches=1, bias_adj=True)
print(cm.estimates)
