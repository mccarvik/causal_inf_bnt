"""
Script for ch9
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
from linearmodels.iv import IV2SLS
import graphviz as gr
import os

style.use("fivethirtyeight")

png_path = "pngs/ch9/"


# Modern IV practice draws a lot of insight from medical sciences. It partitions the world into 4 kinds of subjects, 
# depending on how they respond to the instrumental variable.
# Compliers - stick to what was assigned to them
# Never Takers - refuse to take their medicine. Placebo or new drug
# Always Takers - always takers are those that can somehow get the new drug even if they were assigned to the placebo
# Defiers - take the treatment if assigned to the control and take the control if assigned the treatment. (not that common) 


# let's consider a case where you want to boost user engagement measured by in app purchase. 
# One way to do that, is to come up with a push notification you can use to engage your users.

g = gr.Digraph(format="png")
g.edge("push assigned", "push delivered")
g.edge("push delivered", "in app purchase")
g.edge("income", "in app purchase")
g.edge("income", "push delivered")
g.node("income", color="blue")
g.render(filename=png_path+"push")
os.remove(png_path + "push")

# Up on the causal graph, you have the push assignment. 
# This is random by design, so nothing is causing it. Then, you have a node for if the push was delivered. 
# Not everyone that was assigned to receive the push did it, so you have non compliance here. 
# More specifically, you have some never takers: those that don't receive the treatment even if assigned to it. 
# You also have reasons to suspect that this non compliance is not simply by chance. 
# Since people with older phones are the ones that don't get the push, you can argue that income is also causing push delivery. 
# The richer the person, the more likely it is that he or she has a nicer phone, which in turn makes it more likely that he or she will receive the push. 
# Finally, you have the outcome variable, in app purchase. Keep in mind that we don't know income, so we can't control it.


# To see this, suppose first we have always takers. Some of them will be assigned to the control by chance. 
# But those that are, will take the treatment anyway. This makes them essentially a treated group that is mixed up with the control. 
# As a result of this mix, the causal effect will be harder to find when we have non compliance.
# By the same reasoning, never takers will make those assigned to the treatment look a little bit like the untreated
# because they don't take the treatment even if assigned to it. 
# In this sense, the causal effect of treatment assignment is biased towards zero because non compliance shrinks the detectable impact

g = gr.Digraph(format="png")
g.node("push assigned")
g.edge("push delivered", "in app purchase")
g.edge("income", "in app purchase")
g.edge("income", "push delivered")
g.node("income", color="blue")
g.render(filename=png_path+"random")
os.remove(png_path + "random")

# Local Average Treatment Effect = LATE
# Want to isolate compliers using IVs

# Effect on Engagement to push notifications
data = pd.read_csv("./data/app_engagement_push.csv")
print(data.head())

# First, let's run OLS to see what it would give us.
ols = IV2SLS.from_formula("in_app_purchase ~ 1 + push_assigned + push_delivered", data).fit()
print(ols.summary.tables[1])
# Reasons to believe this is a biased estimate. 
# We know that older phones are having trouble in receiving the push, so, probably, richer customers, with newer phones, are the compliers.
# Since the ones that get the treatment also have more money, we believe this bias is positive and the true impact of the push is lower

# Now, let's try to estimate this effect with Instrumental Variables. First, let's run the first stage.
first_stage = IV2SLS.from_formula("push_delivered ~ 1 + push_assigned", data).fit()
print(first_stage.summary.tables[1])

# Looks like we have a strong first stage.
# Those that get assigned to get the push get it 71.8% of the time. 
# This means that we have something like 28% of never takers. 
# We also have strong reasons to believe there are no always takers, since the intercept parameter is estimated to be zero. 
# This means that no one get's the push if it is not assigned to it.

# run reduced form
reduced_form = IV2SLS.from_formula("in_app_purchase ~ 1 + push_assigned", data).fit()
print(reduced_form.summary.tables[1])

# The reduced form shows that the causal effect of treatment assignment is 2.36. 
# This means that assigning someone to receive the push increases the in-app purchase by 2.36 reais.
# If we divide the reduced form by the first stage, we scale the effect of the instrument by the units of the treatment, we get 3.29
# Running the 2SLS, we get these same estimates, with the bonus of correct standard errors.

iv = IV2SLS.from_formula("in_app_purchase ~ 1 + [push_delivered ~ push_assigned]", data).fit()
print(iv.summary.tables[1])
# 3.29 vs original 27.60
# This makes sense, since the causal effect estimated with OLS is positively biased. 
# We also need to remember about LATE. 3.29 is the average causal effect on compliers. 
# Unfortunately, we can't say anything about those never takers. 
# This means that we are estimating the effect on the richer segment of the population that have newer phones.