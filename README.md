# Prect Student Performance from Game Play

1. For the final submission, I did not use (full data training with 600 fixed epoch) custom metric (macro averaged F1 score).
    a. To maximize the data quantity.
    b. I thought that even though I can find questionwise best epoch with it, the global (all data) maximum score point might be different.
macro f1: average of f1 scores of the true and false labels. Additionally, each time the model calculate it, I sweeped threshold from 0 to 1.0 (0.01 interval).

2. But I could use the macro averaged F1 score function to do feature engineering.
    a. I modified one question's features and see if it affect to the all-question macro f1 score. By including other question models' inference results when calculating the score.
    b. Iterate a and keep only features that contribute to the f1 score.