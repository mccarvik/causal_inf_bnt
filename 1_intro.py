"""
Intro chapter to the causal Inference for the brave and true
"""
import pdb
import pandas as pd
import numpy as np
from scipy.special import expit
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")
png_path = "pngs/ch1/"

np.random.seed(123)
n = 100
tuition = np.random.normal(1000, 300, n).round()
tablet = np.random.binomial(1, expit((tuition - tuition.mean()) / tuition.std())).astype(bool)
enem_score = np.random.normal(200 - 50 * tablet + 0.7 * tuition, 200)
enem_score = (enem_score - enem_score.min()) / enem_score.max()
enem_score *= 1000

data = pd.DataFrame(dict(enem_score=enem_score, Tuition=tuition, Tablet=tablet))
plt.figure(figsize=(6,8))
sns.boxplot(y="enem_score", x="Tablet", data=data).set_title('ENEM score by Tablet in Class')
plt.savefig(png_path + "enem_score.png")
plt.close()

# Y0i is the potential for unit i without the treatment
# Y1i is the potential for unit i with the treatment
# factual --> what actually happened, counterfactual --> potential outcome that didnt happen (ex: what would happen if they received treatment, but they werent treated)
# individual treatment effect --> Y1i - Y0i
# ATE (average treatment effect) --> E(Y1i - Y0i)
# ATET (average treatment effect of treated0 --> E(Y1i - Y0i | T=1)
# The bias term is given by how the treated and control group differ before the treatment, in case neither of them has received the treatment


tabs = pd.DataFrame(dict(
    i= [1,2,3,4],
    y0=[500,600,800,700],
    y1=[450,600,600,750],
    t= [0,0,1,1],
    y= [500,600,600,750],
    te=[-50,0,-200,50],
))
# print(tabs)

plt.figure(figsize=(10,6))
sns.scatterplot(x="Tuition", y="enem_score", hue="Tablet", data=data, s=70).set_title('ENEM score by Tuition Cost')
plt.savefig(png_path + "enem_scores_scatter.png")
plt.close()
