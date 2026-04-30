## Model for tutor prediction

This project based on the Kagle competition. Link to the competition - https://www.kaggle.com/competitions/gb-choose-tutors

The main target of this repository is put into practice the knowledge about Logistic Regression

Description:

In this competition your task will be to predict the probability for a tutor to be a proper one for preparing for the math exam. You will be given two datasets: train.csv (contains all features and the target) and test.csv (only features).

The evaluation metric is ROC-AUC

Competition Rules - You can only use these imports:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn import over_sampling, under_sampling
import matplotlib.pyplot as plt
import seaborn as sns

Dataset Description

Id - unique identifier

age - tutor age

years_of_experience - tutor experience

lesson_price - price for 1 lesson

qualification

physics - teaches physics

chemistry - teaches chemistry

biology - teaches biology

english - teaches english

geography - teaches geography

history - teaches history

mean_exam_points

choose - target feature

As the result: fitted model predicted chosen tutors. Metric ROC-AUC showed 0.824 on Baseline. On test data (the result of the competition) 0.97793
