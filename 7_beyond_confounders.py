"""
Script for ch7
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import graphviz as gr
import pdb
import os
style.use("fivethirtyeight")


png_path = "pngs/ch7/"

data = pd.read_csv("./data/collections_email.csv")
print(data.head())

print("Difference in means:",
      data.query("email==1")["payments"].mean() - data.query("email==0")["payments"].mean())

model = smf.ols('payments ~ email', data=data).fit()
print(model.summary().tables[1])

sns.scatterplot()
sns.scatterplot(x="email", y="payments", 
                alpha=0.8,
                data=data.assign(email=data["email"] + np.random.normal(0, 0.01, size=len(data["email"]))))
plt.plot(np.linspace(-0.2, 1.2), model.params[0] + np.linspace(-1, 2) * model.params[1], c="C1")
plt.xlabel("Email")
plt.ylabel("Payments")
plt.savefig(png_path+"payments.png")
plt.close()

model_email = smf.ols('email ~ credit_limit + risk_score', data=data).fit()
model_payments = smf.ols('payments ~ credit_limit + risk_score', data=data).fit()

residuals = pd.DataFrame(dict(res_payments=model_payments.resid, res_email=model_email.resid))
model_treatment = smf.ols('res_payments ~ res_email', data=residuals).fit()

print("Payments Variance", np.var(data["payments"]))
print("Payments Residual Variance", np.var(residuals["res_payments"]))
print("Email Variance", np.var(data["email"]))
print("Email Residual Variance", np.var(residuals["res_email"]))
print(model_treatment.summary().tables[1])


sns.scatterplot(x="res_email", y="res_payments", data=residuals)
plt.plot(np.linspace(-0.7, 1), model_treatment.params[0] + np.linspace(-1, 2) * model_treatment.params[1], c="C1")
plt.xlabel("Email Residuals")
plt.ylabel("Payments Residuals")
plt.savefig(png_path+"residuals.png")
plt.close()

# Above was done for illustrative reasons. Can just add the controls (credit limit / risk score), with the treatment (email) to obtain the estimates (which are the same as above)
model_2 = smf.ols('payments ~ email + credit_limit + risk_score', data=data).fit()
print(model_2.summary().tables[1])


g = gr.Digraph(format="png")
g.edge("X", "Y"), g.edge("T", "Y")
g.node("T", color="gold")

g.node("email", color="gold")
g.edge("credit_limit", "payments")
g.edge("risk_score", "payments")
g.edge("email", "payments")
g.render(filename=png_path+"conf")
os.remove(png_path + "conf")

hospital = pd.read_csv("./data/hospital_treatment.csv")
print(hospital.head())
hosp_1 = smf.ols('days ~ treatment', data=hospital).fit()
print(hosp_1.summary().tables[1])


# Look at ATE of each hospital individually
hosp_2 = smf.ols('days ~ treatment', data=hospital.query("hospital==0")).fit()
print(hosp_2.summary().tables[1])

hosp_3 = smf.ols('days ~ treatment', data=hospital.query("hospital==1")).fit()
print(hosp_3.summary().tables[1])

# add severity to the model
hosp_4 = smf.ols('days ~ treatment + severity', data=hospital).fit()
print(hosp_4.summary().tables[1])

# The question that arises next is, should we also include hospital in the model? After all, we know that hospitals cause the treatment right? 
# Well, that is true, but once we've controlled for severity, hospital is no longer correlated with the outcome number of days hospitalised. 
# And we know that to be a confounder a variable has to cause both the treatment and the outcome. In this case, we have a variable that only causes the treatment.
# But maybe controlling for it lowers the variance, right? Well, not true again. 
# In order for a control to lower the variance, it has to be a good predictor of the outcome, not of the treatment,

# including it as a confounder can hurt model performance:
hosp_5 = smf.ols('days ~ treatment + severity + hospital', data=hospital).fit()
print(hosp_5.summary().tables[1])
# Adding hospital on top of severity as a control introduced MORE variance to our ATE estimator

# From this formula, we can see that the standard error is inversely proportional to the variance of the variable X
# This means that, if X doesn't change much, it will be hard to estimate its effect on the outcome. This also makes intuitive sense. 
# Take it to the extreme and pretend you want to estimate the effect of a drug, so you conduct a test with 10000 individuals but only 1 of them get the treatment. 
# This will make finding the ATE very hard, we will have to rely on comparing a single individual with everyone else. 
# Another way to say this is that we need lots of variability in the treatment to make it easier to find its impact.

# As to why including hospitals in the model increases the error of our estimate, it is because it is a good predictor of the treatment and not of the outcome 
# (once we control for severity). So, by predicting the treatment, it effectively makes it so that it's variance is lower

model_treatment = smf.ols('treatment ~ severity + hospital', data=hospital).fit()
model_days = smf.ols('days ~ severity + hospital', data=hospital).fit()
residuals = pd.DataFrame(dict(res_days=model_days.resid, res_treatment=model_treatment.resid))
model_treatment = smf.ols('res_days ~ res_treatment', data=residuals).fit()
print(model_treatment.summary().tables[1])
print("Treatment Variance", np.var(hospital["treatment"]))
print("Treatment Residual Variance", np.var(residuals["res_treatment"]))
sigma_hat = sum(model_treatment.resid**2)/(len(model_treatment.resid)-2)
var = sigma_hat/sum((residuals["res_treatment"] - residuals["res_treatment"].mean())**2)
print("SE of the Coeficient:", np.sqrt(var))

# So the bottom line is that we should add controls that are both correlated with the treatment and the outcome (confounder), like the severity in the model above. 
# We should also add controls that are good predictors of the outcome, even if they are not confounders, because they lower the variance of our estimates. 
# However, we should NOT add controls that are just good predictors of the treatment, because they will increase the variance of our estimates.
# Here is a picture of what this situation looks like with causal graphs.

g = gr.Digraph(format="png")
g.edge("X", "T"), g.edge("T", "Y")
g.node("T", color="gold")
g.node("treatment", color="gold")
g.edge("severity", "hospital")
g.edge("severity", "days")
g.edge("hospital", "treatment")
g.edge("treatment", "days")
g.render(filename=png_path+"controls")
os.remove(png_path + "controls")


# Bad Controls
email_1 = smf.ols('payments ~ email + credit_limit + risk_score', data=data).fit()
print(email_1.summary().tables[1])

email_2 = smf.ols('payments ~ email + credit_limit + risk_score + opened + agreement', data=data).fit()
print(email_2.summary().tables[1])

g = gr.Digraph(format="png")
g.edge("email", "payments")
g.edge("email", "opened")
g.edge("email", "agreement")
g.edge("opened", "payments")
g.edge("opened", "agreement")
g.edge("agreement", "payments")

g.edge("credit_limit", "payments")
g.edge("credit_limit", "opened")
g.edge("credit_limit", "agreement")
g.edge("risk_score", "payments")
g.edge("risk_score", "opened")
g.edge("risk_score", "agreement")
g.render(filename=png_path+"bad_controls")
os.remove(png_path + "bad_controls")
# If you think about it, opened and agreement are surely correlated with the email. 
# After all, you can't open the email if you didn't receive it and we've also said that the agreement only considers renegotiation that happened after the email has been sent. 
# But they don't cause email! Instead, they are caused by it!

# selection bias is when we control for a common effect or a variable in between the path from cause to effect. 
# As a rule of thumb, always include confounders and variables that are good predictors of Y in your model
# Always exclude variables that are:
# - good predictors of only T, 
# - mediators between the treatment and outcome
# - or common effect of the treatment and outcome

# The Bad COP
# COP = Conditioned on Participation
# You have a continuous variable that you want to predict but its distribution is overrepresented at zero. 
# For instance, if you want to model consumer spending, you will have something like a gamma distribution, but with lots of zeros.
plt.hist(np.concatenate([
    np.random.gamma(5, 50, 1000), 
    np.zeros(700)
]), bins=20)
plt.xlabel("Customer Spend")
plt.title("Distribution Customer Spend");
plt.savefig(png_path + "gamma.png")
plt.close()

# Example:
# Let's say that we want to estimate how a marketing campaign increases how much people spend on our product. 
# This marketing campaign has been randomized, so we don't need to worry about confounding. 
# In this example, we can break up the customers into two segments. 
# First, there are those that will only buy our products if they see a marketing campaign. Let's call these customers the frugal ones. They don't spend unless we give them an extra push. 
# Then there are the customers that will spend even without the campaign. The campaign makes them spend more, but they would already spend without seeing it anyway. Let's call them the rich customers

# To estimate the ATE of the campaign, since we have randomization, all we need to do is compare the treated with the untreated. 
# But, suppose we use the COP formulation where we break up estimation into two models, a participation model and the COP 
# This removes everyone that didn't spend from the analysis.

# When we do that, the treated and control are no longer comparable. 
# As we can see, the untreated is now only composed of the segment of customers that will spend even without the campaign.
# We are comparing those that received the marketing: (those that now spend + those that spend a little more)
# with those that did not receive the marketing: (those that were gonna spend anyway)
# and leaving out those that did not receive the marketing but didnt spend

# summary, here is what bad control looks like
g = gr.Digraph(format="png")
g.edge("T", "X_1"), g.node("T", color="gold"), g.edge("X_1", "Y"), g.node("X_1", color="red")
g.edge("T", "X_2"), g.edge("Y", "X_2"), g.node("X_2", color="red")
g.render(filename=png_path+"bad_controls2")
os.remove(png_path + "bad_controls2")
