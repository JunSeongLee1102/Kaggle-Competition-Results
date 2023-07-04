# Prect Student Performance from Game Play
link: https://www.kaggle.com/competitions/predict-student-performance-from-game-play

## Goal of the Competition
The goal of this competition is to predict student performance during game-based learning in real-time. You'll develop a model trained on one of the largest open datasets of game logs.

## Basic description
There are total 18 questions and the log data of the each game player exist.   
Target: Whether the game players' answer for the each question is correct or not.

## My solution
1. For the final submission, I did not use (full data training with 600 fixed epoch) custom metric (macro averaged F1 score).  
    a. To maximize the data quantity.  
    b. I thought that even though I can find questionwise best epoch with it, the global (all data) maximum score point might be different.  
macro f1: average of f1 scores of the true and false labels. Additionally, each time the model calculate it, I sweeped threshold from 0 to 1.0 (0.01 interval).  
  
2. But I could use the macro averaged F1 score function to do feature engineering.  
    a. I modified one question's features and see if it affect to the all-question macro f1 score. By including other question models' inference results when calculating the score.  
    b. Iterate a and keep only features that contribute to the f1 score.  
  
## Install dependencies
conda env update --file prediction_environment.yml 

## Download inputs
Download input files from the following links and put them in the main directory of the 01_Predict_Student_Performance_from_Game_Play  
Main dataset: https://www.kaggle.com/competitions/predict-student-performance-from-game-play   
Features (from Vadim Kamaev): https://www.kaggle.com/datasets/vadimkamaev/featur  
folder structure (make inputs directory and put them in the directory)
-main/inputs/featur  
-main/inputs/predict-student-performance-from-game-play

## Order
1. Set the input directory path: 
2. Generate dataset by run catboost_cudf_questionwise_train.ipynb (It will display error because the fold is not splited).  
3. Generate data split by data_split.ipynb  
4. Train the catboost model by run.
