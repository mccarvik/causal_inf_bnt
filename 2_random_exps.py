"""
Script for chapter 2 -> randomized experiments
"""

#  first tool we have to make the bias vanish: Randomised Experiments.
# Randomised experiments randomly assign individuals in a population to a treatment or to a control group

import pandas as pd
import numpy as np

data = pd.read_csv("./data/online_classroom.csv")
print(data.shape)
print(data.head())

print(
(data
 .assign(class_format = np.select(
     [data["format_ol"].astype(bool), data["format_blended"].astype(bool)],
     ["online", "blended"],
     default="face_to_face"
 ))
 .groupby(["class_format"])
 .mean())
)

# In a randomised experiment, the mechanism that assigns units to one treatment or the other is, well, random. 
# As we will see later, all causal inference techniques will somehow try to identify the assignment mechanisms of the treatments. 
# When we know for sure how this mechanism behaves, causal inference will be much more confident, even if the assignment mechanism isn't random.
