"""
Script for chapter 4
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pylab
import pandas as pd
import numpy as np
import graphviz as gr
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
style.use("fivethirtyeight")

png_path = "pngs/ch4/"

g = gr.Digraph(format="png")
g.edge("Z", "X")
g.edge("U", "X")
g.edge("U", "Y")

g.edge("medicine", "survived")
g.edge("severeness", "survived")
g.edge("severeness", "medicine")
g.render(filename=png_path+"graph")
os.remove(png_path + "graph")

g = gr.Digraph(format="png")
g.edge("A", "B")
g.edge("B", "C")

g.edge("X", "Y")
g.edge("Y", "Z")
g.node("Y", "Y", color="red")


g.edge("causal knowledge", "solve problems")
g.edge("solve problems", "job promotion")
g.render(filename=png_path+"graph2")
os.remove(png_path + "graph2")


g = gr.Digraph(format="png")
g.edge("C", "A")
g.edge("C", "B")

g.edge("X", "Y")
g.edge("X", "Z")
g.node("X", "X", color="red")

g.edge("statistics", "causal inference")
g.edge("statistics", "machine learning")
g.render(filename=png_path+"graph3")
os.remove(png_path + "graph3")

g = gr.Digraph(format="png")
g.edge("B", "C")
g.edge("A", "C")

g.edge("Y", "X")
g.edge("Z", "X")
g.node("X", "X", color="red")

g.edge("statistics", "job promotion")
g.edge("flatter", "job promotion")
g.render(filename=png_path+"graph4")
os.remove(png_path + "graph4")


g = gr.Digraph(format="png")
g.edge("C", "A")
g.edge("C", "B")
g.edge("D", "A")
g.edge("B", "E")
g.edge("F", "E")
g.edge("A", "G")
g.render(filename=png_path+"graph5")
os.remove(png_path + "graph5")


# The first significant cause of bias is confounding. It happens when the treatment and the outcome share a common cause. 
# We need to close all backdoor paths between the treatment and the outcome to identify the causal effect
# In this case, control for X / intelligence
g = gr.Digraph(format="png")
g.edge("X", "T")
g.edge("X", "Y")
g.edge("T", "Y")

g.edge("Intelligence", "Educ"),
g.edge("Intelligence", "Wage"),
g.edge("Educ", "Wage")
g.render(filename=png_path+"confounding_bias")
os.remove(png_path + "confounding_bias")


# Unfortunately, it is not always possible to control all common causes. Sometimes, there are unknown causes or known causes that we can't measure. 
# The case of intelligence is one of the latter. 
# Sssume for a moment that intelligence can't affect your education directly
# It affects how well you do on the SATs, but the SATs determine your level of schooling since it opens the possibility of a good college. 
# Even if we can't control for the unmeasurable intelligence, we can control for SAT and close that backdoor path.

g = gr.Digraph(format="png")
g.edge("X1", "T")
g.edge("T", "Y")
g.edge("X2", "T")
g.edge("X1", "Y")
g.edge("U", "X2")
g.edge("U", "Y")

g.edge("Family Income", "Educ")
g.edge("Educ", "Wage")
g.edge("SAT", "Educ")
g.edge("Family Income", "Wage")
g.edge("Intelligence", "SAT")
g.edge("Intelligence", "Wage")
g.render(filename=png_path+"conf_unmeas")
os.remove(png_path + "conf_unmeas")

# What if the unmeasured variable causes the treatment and the outcome directly? 
# In the following example, intelligence causes education and income now. So there is confounding in the relationship between the treatment education and the outcome wage. 
# In this case, we can't control the confounder because it is unmeasurable. 
# However, we have other measured variables that can act as a proxy for the confounder. 
# Those variables are not in the backdoor path, but controlling for them will help lower the bias (but it won't eliminate it). 
# Those variables are sometimes referred to as surrogate confounders.

# In our example, we can't measure intelligence, but we can measure some of its causes, like the father's and mother's education
# and some of its effects, like IQ or SAT score. Controlling for those surrogate variables is not sufficient to eliminate bias, but it helps.

g = gr.Digraph(format="png")
g.edge("X", "U")
g.edge("U", "T")
g.edge("T", "Y")
g.edge("U", "Y")

g.edge("Intelligence", "IQ")
g.edge("Intelligence", "SAT")
g.edge("Father's Educ", "Intelligence")
g.edge("Mother's Educ", "Intelligence")

g.edge("Intelligence", "Educ")
g.edge("Educ", "Wage")
g.edge("Intelligence", "Wage")
g.render(filename=png_path+"conf_unmeas_2")
os.remove(png_path + "conf_unmeas_2")


# If confounding bias happens when we don't control for a common cause, selection bias is more related to effects. 
# Often, selection bias arises when we control for more variables than we should.
# to be sure you won't have confounding, you control for many variables. Among them, you control for investments
# But investment is not a common cause of education and wage. Instead, it is a consequence of both. 
# More educated people both earn more and invest more. Also, those who make more invest more. 
# Since investment is a collider, by conditioning on it, you are opening a second path between the treatment and the outcome, 
# which will make it harder to measure the direct effect. 
# One way to think about this is that by controlling investments, you look at small groups of the population where investment is the same 
# and then find the effect of education on those groups. 
# But by doing so, you are also indirectly and inadvertently not allowing wages to change much. 
# As a result, you won't be able to see how education changes wages because you are not allowing wages to change as they should.

g = gr.Digraph(format = "png")
g.edge("T", "X")
g.edge("T", "Y")
g.edge("Y", "X")
g.node("X", "X", color="red")

g.edge("Educ", "Investments")
g.edge("Educ", "Wage")
g.edge("Wage", "Investments")

g.render(filename=png_path+"select_bias")
os.remove(png_path + "select_bias")


# if someone invests, knowing that they have high education explains away the second cause, which is wage. 
# Conditioned on investing, higher education is associated with low wages and we have a negative bias
g = gr.Digraph(format="png")
g.edge("T", "X")
g.edge("T", "Y")
g.edge("Y", "X")
g.edge("X", "S")
g.node("S", "S", color="red")
g.render(filename=png_path+"select_bias2")
os.remove(png_path + "select_bias2")


# A similar thing happens when we condition on a mediator of the treatment. 
# A mediator is a variable between the treatment and the outcome. 
# It mediates the causal effect. For example, you decide to control whether or not the person had a white-collar job
# Once again, this conditioning biasses the causal effect estimation. 
# This time, not because it opens a front door path with a collider, but because it closes one of the channels through which the treatment operates. 
# In our example, getting a white-collar job is one way more education leads to higher pay. 
# By controlling it, we close this channel and leave open only the direct effect of education on wages.


g = gr.Digraph(format="png")
g.edge("T", "X")
g.edge("T", "Y")
g.edge("X", "Y")
g.node("X", "X", color="red")

g.edge("educ", "white collar")
g.edge("educ", "wage")
g.edge("white collar", "wage")
g.render(filename=png_path+"mediator")
os.remove(png_path + "mediator")


