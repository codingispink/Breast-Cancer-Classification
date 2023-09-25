# Breast-Cancer-Classification
**Goal of the Project:** The project analyzes 570 breast cancer cases. Based on 29 features, the project employs KNN and decision tree to classify tumor types (Malignant or Benign).

## Summary
I. The Importance of The Project

II. KNN
  1. Preparation
  2. Train the CNN model
  3. Apply KNN Model to make prediction
  4. Confusion Matrix
     
III. Decision Tree
  1. Train a Decision Tree
  2. Confusion Matrix
  3. Choose the probability cutoff
  4. New Confusion Matrix

### I. The importance of the Project
According to American Cancer Society, **1 in 8 women** will be diagnosed with breast cancer in her lifetime. In 2023, nearly **300,000** women and **3,000** men are diagnosed with invasive breast cancers. However, if caught in its earliest and localized stages, the survival rate for breast cancer could be up to 99%. Therefore, the breast cancer classification project can be important in helping everyone, espcially women, navigating through their health journey.

### II. KNN 
#### 1. Preparation:
These are the required packages:
```
install.packages("caret") 
install.packages("rpart") 
install.packages("rpart.plot")
library("caret") 
library("rpart")
library("rpart.plot")
install.packages(rpart.plot)
library(rpart.plot)
```

**Data Normalization:**
```
data_stand <- data
data_stand[,2:31] <-apply(data_stand[,2:31], 2, scale)
```

Then, randomly split the dataset into training and testing dataset. We will split it into 80% training and 20% testing. To ensure that our random samples are the same, please set the random seed to 131 using set.seed(166) before data partition.

```
set.seed(166)
train_rows <- createDataPartition(y = data_stand$diagnosis, p =0.8, list = FALSE)
head(train_rows, n=10)
```
Then, use the data partition results to create the testing and training data.

```
data_train <-data[train_rows,]
data_test <-data[-train_rows,]
data_stand_train <- data_stand[train_rows,]
data_stand_test <- data_stand[-train_rows,]
```

#### 2. Train the CNN model:
We can train the model with function train(function, data, method).
```
fitKNN <- train(Churn.~., data=data_stand_train, method = "knn")
fitKNN
```
The results:

```
k-Nearest Neighbors 

456 samples
 30 predictor
  2 classes: 'B', 'M' 

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 456, 456, 456, 456, 456, 456, ... 
Resampling results across tuning parameters:

  k  Accuracy   Kappa    
  5  0.9581046  0.9102827
  7  0.9621953  0.9185721
  9  0.9646111  0.9239263

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was k = 9.
```
The algorithm automatically tries three k values, 5,7,9, and selects the k=9 based on the largest accuracy.

**Control the model tuning**:

Many models have tunable parameters, such as k in kNN models. Through tuneGrid, the model tries on different parameter combinations and returns the best one. We need to build and provide a parameter grid in this case. The grid is often built through expand.grid(). Set a custom parameter value via trControl, which allows you to set various training parameters such as , (whether to return class probabilities along with predicted value). The parameters are specified using the function trainControl. 

```
ctrl = trainControl(search="grid")
grid = expand.grid(k=c(3,5,7,9))
fitKNN <- train(diagnosis~., data=data_stand_train, method="knn", trControl=ctrl, tuneGrid=grid)
fitKNN
```

**Modify the training parameters:**

Use the default training control parameters by removing trControl = ctrl. Explore alternatives value of k (5, 7, 9, 11, 13). 
```
grid=expand.grid(k=c(5,7,9,11,13))
fitKNN <- train(diagnosis~., data=data_stand_train, method="knn", tuneGrid=grid)
fitKNN
```
**Plot the Training Model**

```
ggplot(fitKNN)
```
![000012](https://github.com/codingispink/Breast-Cancer-Classification/assets/138828365/63afb758-306c-4c47-b2af-4baaf4beb78a)

You can notice that the elbow is at k=9.

#### 3. Apply the kNN model to make new predictions

We are going to use the model we built on the testing data which the model has not seen. We can then compare the predictions of the model with the real value to evaluate 
how good our model is. 

```
knn_predictions <- predict(fitKNN, data_stand_test)
head(knn_predictions)
```

#### 4. Evaluate the model performance

For this classification project, we will use the confusionMatrix to evaluate the results.
```
confusionMatrix(knn_predictions, as.factor(data_stand_test$diagnosis), positive = "M")
```
**The results**: This confusion matrix reports on accuracy, sensitivity and specificity. However, some other metrics that are very important but we don't see here are precision and recall. We need to modify the confusion matrix to add precision and recall.
```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 71  7
         M  0 35
                                          
               Accuracy : 0.9381          
                 95% CI : (0.8765, 0.9747)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : 1.718e-14       
                                          
                  Kappa : 0.8627          
                                          
 Mcnemar's Test P-Value : 0.02334         
                                          
            Sensitivity : 0.8333          
            Specificity : 1.0000          
         Pos Pred Value : 1.0000          
         Neg Pred Value : 0.9103          
             Prevalence : 0.3717          
         Detection Rate : 0.3097          
   Detection Prevalence : 0.3097          
      Balanced Accuracy : 0.9167          
                                          
       'Positive' Class : M
```
**Modify the Confusion Matrix:**
```
confusionMatrix(knn_predictions, as.factor(data_stand_test$diagnosis), mode = "prec_recall", positive = "M")
```
The results now include precision and recall

```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 71  7
         M  0 35
                                          
               Accuracy : 0.9381          
                 95% CI : (0.8765, 0.9747)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : 1.718e-14       
                                          
                  Kappa : 0.8627          
                                          
 Mcnemar's Test P-Value : 0.02334         
                                          
              Precision : 1.0000          
                 Recall : 0.8333          
                     F1 : 0.9091          
             Prevalence : 0.3717          
         Detection Rate : 0.3097          
   Detection Prevalence : 0.3097          
      Balanced Accuracy : 0.9167          
                                          
       'Positive' Class : M
```

### Decision Tree:
First, we can **split the original data**:

```
data_train <- data[train_rows,]
data_test <- data[train_rows,]
```
#### Train the decision model tree:
``` 
set.seed(42)
fitDT <- train(diagnosis~., data = data_train, method = "rpart")
```
#### View the decision tree:

```
library(rpart)
fitDT$finalModel
```
![000010](https://github.com/codingispink/Breast-Cancer-Classification/assets/138828365/47c97931-7d14-437f-b3e8-a97f71b85d92)

 **Interpreting the Decision Tree:**
Out of all cases observed in the dataset, around 37% of the tumor is predicted to be benign. In these benign tumor cases, if the area_worst is >=885, there is a 96% chance the tumor is malignant. Otherwise, there is a 9% chance that the tumor is benign. Of these benign tumors, those with concave.points_worst <0.16 has a 3% chance of the tumors being benign, and those with concave.points_worst >=0.16 has a 90% chance of the tumors being malignant.

#### Make Predictions Using the Decision Tree Model:
Predict the tumor type for the new data and the predicted class only.

```
DT_predictions <- predict(fitDT$finalModel, newdata = data_test, type = "class")
(DT_predictions)
```
**Confusion Matrix:**
```
confusionMatrix(DT_predictions, as.factor(data_test$diagnosis), mode = "prec_recall", positive = "M")
```
The result:
```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 69  8
         M  2 34
                                          
               Accuracy : 0.9115          
                 95% CI : (0.8433, 0.9567)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : 6.062e-12       
                                          
                  Kappa : 0.8048          
                                          
 Mcnemar's Test P-Value : 0.1138          
                                          
              Precision : 0.9444          
                 Recall : 0.8095          
                     F1 : 0.8718          
             Prevalence : 0.3717          
         Detection Rate : 0.3009          
   Detection Prevalence : 0.3186          
      Balanced Accuracy : 0.8907          
                                          
       'Positive' Class : M           
```

The results show a recall of 80% and a precision of 94%. In medical diagnosis cases like this, false-negative cases are a crime. Therefore, we care more about increasing recall rate. Let's increase the recall rate.

#### Choose your own probability cutoff
The predict() function can return predicted probabilities, which gives you a chance to apply your own cut-off (by default it is 50%).
First, we can get the predicted probabilities.
```
DT_prob<- as.data.frame(predict(fitDT$finalModel, newdata = data_test, type = "prob"))
head(DT_prob)
```
Second, create predicted class using a custom-cutoff.
``` 
DT_prob$pred_class <- ifelse(DT_prob$M > 0.4, "M", "B")
```
#### New Confusion Matrix
```
confusionMatrix(as.factor(DT_prob$pred_class), as.factor(data_test$diagnosis),mode = "prec_recall", positive = "M")
```
The result:
```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 62  1
         M  9 41
                                          
               Accuracy : 0.9115          
                 95% CI : (0.8433, 0.9567)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : 6.062e-12       
                                          
                  Kappa : 0.8176          
                                          
 Mcnemar's Test P-Value : 0.02686         
                                          
              Precision : 0.8200          
                 Recall : 0.9762          
                     F1 : 0.8913          
             Prevalence : 0.3717          
         Detection Rate : 0.3628          
   Detection Prevalence : 0.4425          
      Balanced Accuracy : 0.9247          
                                          
       'Positive' Class : M               
```
 In this new confusion matrix, you notice that the recall rate is now up to 98%. This is what we would like to see.

#### Conclusion

Based on the 2 confusion matrixes of the KNN and decision tree, we can see that the recall of the decision tree is higher than that of the kNN model (98% compared to 83%). Therefore, decision tree in this case is the better model for classifying tumor types.

 ### ACKNOWLEDGEMENT
 [1] Classification Lecture, Professor De Liu at the University of Minnesota (IDSC 4444)
 
 [2] American Cancer Society, https://www.cancer.org/cancer/types/breast-cancer/about/how-common-is-breast-cancer.html
 
 [3] Precision vs Recall: Differences, Use Cases and Evaluation: https://www.v7labs.com/blog/precision-vs-recall-guide#:~:text=In%20most%20high%2Drisk%20disease,the%20correctness%20of%20our%20model.

 [4] Breast Cancer Prediction Data Set: https://www.kaggle.com/code/buddhiniw/breast-cancer-prediction
