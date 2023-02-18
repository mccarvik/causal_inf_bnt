import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

png_path = "pngs/ch6/"

style.use("fivethirtyeight")
np.random.seed(876)
enem = pd.read_csv("./data/enem_scores.csv").sample(200)
plt.figure(figsize=(8,4))
sns.scatterplot(y="avg_score", x="number_of_students", data=enem)
sns.scatterplot(y="avg_score", x="number_of_students", s=100, label="Trustworthy",
                data=enem.query(f"number_of_students=={enem.number_of_students.max()}"))
sns.scatterplot(y="avg_score", x="number_of_students", s=100, label="Not so Much",
                data=enem.query(f"avg_score=={enem.avg_score.max()}"))
plt.title("ENEM Score by Number of Students in the School");
plt.savefig(png_path + "enem1.png")
plt.close()

# phenomenon of having a region of low variance and another of high variance is called heteroskedasticity
# heteroskedasticity is when the variance is not constant across all values of the features

wage = pd.read_csv("./data/wage.csv")[["wage", "lhwage", "educ", "IQ"]]
print(wage.head())
model_1 = smf.ols('lhwage ~ educ', data=wage).fit()
print(model_1.summary().tables[1])

group_wage = (wage
              .assign(count=1)
              .groupby("educ")
              .agg({"lhwage":"mean", "count":"count"})
              .reset_index())
print(group_wage)

# weighting the groups
model_2 = smf.wls('lhwage ~ educ', data=group_wage, weights=group_wage["count"]).fit()
print(model_2.summary().tables[1])

# if we didnt weight the group
model_3 = smf.ols('lhwage ~ educ', data=group_wage).fit()
print(model_3.summary().tables[1])

sns.scatterplot(x="educ", y = "lhwage", size="count", legend=False, data=group_wage, sizes=(40, 400))
plt.plot(wage["educ"], model_2.predict(wage["educ"]), c="C1", label = "Weighted")
plt.plot(wage["educ"], model_3.predict(wage["educ"]), c="C2", label = "Non Weighted")
plt.xlabel("Years of Education")
plt.ylabel("Log Hourly Wage")
plt.legend()
plt.savefig(png_path+"weighted.png")
plt.close()

group_wage = (wage
              .assign(count=1)
              .groupby("educ")
              .agg({"lhwage":"mean", "IQ":"mean", "count":"count"})
              .reset_index())

# added other covariates
model_4 = smf.wls('lhwage ~ educ + IQ', data=group_wage, weights=group_wage["count"]).fit()
print("Number of observations:", model_4.nobs)
print(model_4.summary().tables[1])

wage = (pd.read_csv("./data/wage.csv")
        .assign(hwage=lambda d: d["wage"] / d["hours"])
        .assign(T=lambda d: (d["educ"] > 12).astype(int)))
print(wage[["hwage", "IQ", "T"]].head())
print(smf.ols('hwage ~ T', data=wage).fit().summary().tables[1])

m = smf.ols('hwage ~ T+IQ', data=wage).fit()
plt_df = wage.assign(y_hat = m.fittedvalues)

plt.plot(plt_df.query("T==1")["IQ"], plt_df.query("T==1")["y_hat"], c="C1", label="T=1")
plt.plot(plt_df.query("T==0")["IQ"], plt_df.query("T==0")["y_hat"], c="C2", label="T=0")
plt.title(f"E[T=1|IQ] - E[T=0|IQ] = {round(m.params['T'], 2)}")
plt.ylabel("Wage")
plt.xlabel("IQ")
plt.legend()
plt.savefig(png_path + "wage_bump.png")
plt.close()

m = smf.ols('hwage ~ T*IQ', data=wage).fit()
plt_df = wage.assign(y_hat = m.fittedvalues)

plt.plot(plt_df.query("T==1")["IQ"], plt_df.query("T==1")["y_hat"], c="C1", label="T=1")
plt.plot(plt_df.query("T==0")["IQ"], plt_df.query("T==0")["y_hat"], c="C2", label="T=0")
plt.title(f"E[T=1|IQ] - E[T=0|IQ] = {round(m.params['T'], 2)}")
plt.ylabel("Wage")
plt.xlabel("IQ")
plt.legend()
plt.savefig(png_path +"inter_term.png")
plt.close()

# First added term tells us how much IQ increases wages for the non-treated. So, in our case, it is something like 0.11. 
# This means that for each 1 extra IQ point, the person that has not completed 12th grade should expect to gain an extra 11 cents per hour. 
# Adding interaction term: It tells us how much IQ increases the effect of graduating 12th grade. 
# In our case, this parameter is 0.024, which means that for each extra IQ point, graduating 12th grade gives 2 extra cents. 
# This might not seem much, but compare someone with 60IQ and with 140IQ. The first one will get an increase of 1.44 in wage (60 * 0.024), 
# while the person with 140 IQ will gain an extra 3.36 dollars (140 * 0.024) when graduating from 12th grade.

# In simple modeling jargon, this interaction term allows the treatment effect to change by levels of the features (only IQ, in this example) 
# The result is that if we plot the prediction lines, we will see that they are no longer parallel
# and that those that graduate 12th grade (T=1) have a higher slope on IQ, higher IQ benefit more from graduating than lower IQ. 
# This is sometimes referenced as effect modification or heterogeneous treatment effect.

wage_ed_bins = (wage
                .assign(IQ_bins = lambda d: pd.qcut(d["IQ"], q=4, labels=range(4)))
                [["hwage", "educ", "IQ_bins"]])
print(wage_ed_bins.head())

model_dummy = smf.ols('hwage ~ C(educ)', data=wage).fit()
print(model_dummy.summary().tables[1])

plt.scatter(wage["educ"], wage["hwage"])
plt.plot(wage["educ"].sort_values(), model_dummy.predict(wage["educ"].sort_values()), c="C1")
plt.xlabel("Years of Education")
plt.ylabel("Hourly Wage");
plt.savefig(png_path+"all_dummies.png")
plt.close()

# this removes any assumption about the functional form of how education affects wages. 
# We don't need to worry about logs anymore. In essence, this model is completely non-parametric. 

t1 = wage.query("educ==17")["hwage"]
t0 = wage.query("educ==9")["hwage"]
print("E[Y|T=9]:", t0.mean())
print("E[Y|T=17]-E[Y|T=9]:", t1.mean() - t0.mean())

model_dummy_2 = smf.ols('hwage ~ C(educ) + C(IQ_bins)', data=wage_ed_bins).fit()
print(model_dummy_2.summary().tables[1])
