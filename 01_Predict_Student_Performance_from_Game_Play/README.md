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

