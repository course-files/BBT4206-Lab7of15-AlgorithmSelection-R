# *****************************************************************************
# Lab 5: Model Selection ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# The choice of evaluation metric depends on the specific problem,
# the characteristics of the data, and the goals of the modeling task.
# It's often a good practice to use multiple evaluation metrics to gain a more
# comprehensive understanding of a model's performance.

# There are several evaluation metrics that can be used to evaluate algorithms.
# The default metrics used are:
## (1) "Accuracy" for classification problems and
## (2) "RMSE" for regression problems

# Accuracy is the percentage of correctly classified instances out of all
# instances. Accuracy is more useful in binary classification problems than
# in multi-class classification problems.

# On the other hand, Cohen's Kappa is similar to Accuracy however, it is more
# useful on classification problems that do not have an equal distribution of
# instances amongst the classes in the dataset.

# For example, instead of Red are 50 instances and Blue are 50 instances,
# the distribution can be that Red are 70 instances and Blue are 30 instances.

# STEP 1. Install and Load the Required Packages ----
## stats ----
if (require("stats")) {
  require("stats")
} else {
  install.packages("stats", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## MASS ----
if (require("MASS")) {
  require("MASS")
} else {
  install.packages("MASS", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# There are hundreds of algorithms to choose from.
# A list of the classification and regression algorithms offered by
# the caret package can be found here:
# http://topepo.github.io/caret/available-models.html

# The goal of predictive modelling is to use the most appropriate algorithm to
# design an accurate model that represents the dataset. Selecting the most
# appropriate algorithm is a process that involves trial-and-error.

# If the most appropriate algorithm was known beforehand, then it would not be
# necessary to use Machine Learning. The trial-and-error approach to selecting
# the most appropriate algorithm involves evaluating a diverse set of
# algorithms on the dataset, and identifying the algorithms that create
# accurate models and the ones that do not.

# Once you have a shortlist of the top algorithms, you can then improve their
# results further by either tuning the algorithm parameters or by combining the
# predictions of multiple models using ensemble methods.

# A. Linear Algorithms ----
## 1.a. Linear Regression using Ordinary Least Squares without caret ----
# The lm() function is in the stats package and creates a linear regression
# model using ordinary least squares (OLS).

### Load and split the dataset ----
data(BostonHousing)

# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(BostonHousing$medv,
                                   p = 0.8,
                                   list = FALSE)
boston_housing_train <- BostonHousing[train_index, ]
boston_housing_test <- BostonHousing[-train_index, ]

### Train the model ----
boston_housing_model_lm <- lm(medv ~ ., boston_housing_train)

### Display the model's details ----
print(boston_housing_model_lm)

### Make predictions ----
predictions <- predict(boston_housing_model_lm, boston_housing_test[, 1:13])

### Display the model's evaluation metrics ----
#### RMSE ----
rmse <- sqrt(mean((boston_housing_test$medv - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

#### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((boston_housing_test$medv - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

#### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((boston_housing_test$medv - mean(boston_housing_test$medv))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

#### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

#### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - boston_housing_test$medv)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

## 1.b. Linear Regression using Ordinary Least Squares with caret ----
### Load and split the dataset ----
data(BostonHousing)

# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(BostonHousing$medv,
                                   p = 0.8,
                                   list = FALSE)
boston_housing_train <- BostonHousing[train_index, ]
boston_housing_test <- BostonHousing[-train_index, ]

### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
boston_housing_caret_model_lm <- train(medv ~ ., data = boston_housing_train,
                                       method = "lm", metric = "RMSE",
                                       preProcess = c("center", "scale"),
                                       trControl = train_control)

### Display the model's details ----
print(boston_housing_caret_model_lm)

### Make predictions ----
predictions <- predict(boston_housing_caret_model_lm,
                       boston_housing_test[, 1:13])

### Display the model's evaluation metrics ----
#### RMSE ----
rmse <- sqrt(mean((boston_housing_test$medv - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

#### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((boston_housing_test$medv - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

#### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((boston_housing_test$medv - mean(boston_housing_test$medv))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

#### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

#### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - boston_housing_test$medv)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

## 2.a. Logistic Regression without caret ----
# The glm() function is in the stats package and creates a
# generalized linear model for regression or classification.
# It can be configured to perform a logistic regression suitable for binary
# classification problems.

### Load and split the dataset ----
data(PimaIndiansDiabetes)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.7,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

### Train the model ----
diabetes_model_glm <- glm(diabetes ~ ., data = pima_indians_diabetes_train,
                          family = binomial(link = "logit"))

### Display the model's details ----
print(diabetes_model_glm)

### Make predictions ----
probabilities <- predict(diabetes_model_glm, pima_indians_diabetes_test[, 1:8],
                         type = "response")
print(probabilities)
predictions <- ifelse(probabilities > 0.5, "pos", "neg")
print(predictions)

### Display the model's evaluation metrics ----
table(predictions, pima_indians_diabetes_test$diabetes)

# Read the following article on how to compute various evaluation metrics using
# the confusion matrix:
# https://en.wikipedia.org/wiki/Confusion_matrix

## 2.b. Logistic Regression with caret ----
### Load and split the dataset ----
data(PimaIndiansDiabetes)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.7,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

### Train the model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)
# We can use "regLogistic" instead of "glm"
# Notice the data transformation applied when we call the train function
# in caret, i.e., a standardize data transform (centre + scale)
set.seed(7)
diabetes_caret_model_logistic <-
  train(diabetes ~ ., data = pima_indians_diabetes_train,
        method = "regLogistic", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)

### Display the model's details ----
print(diabetes_caret_model_logistic)

### Make predictions ----
predictions <- predict(diabetes_caret_model_logistic,
                       pima_indians_diabetes_test[, 1:8])

### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         pima_indians_diabetes_test[, 1:9]$diabetes)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 3.a.  Linear Discriminant Analysis without caret ----
# The lda() function is in the MASS package and creates a linear model of a
# multi-class classification problem.

### Load and split the dataset ----
data(PimaIndiansDiabetes)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.7,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

### Train the model ----
diabetes_model_lda <- lda(diabetes ~ ., data = pima_indians_diabetes_train)

### Display the model's details ----
print(diabetes_model_lda)

### Make predictions ----
predictions <- predict(diabetes_model_lda,
                       pima_indians_diabetes_test[, 1:8])$class

### Display the model's evaluation metrics ----
table(predictions, pima_indians_diabetes_test$diabetes)

# Read the following article on how to compute various evaluation metrics using
# the confusion matrix:
# https://en.wikipedia.org/wiki/Confusion_matrix

## 3.b.  Linear Discriminant Analysis with caret ----
### Load and split the dataset ----
data(PimaIndiansDiabetes)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.7,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

### Train the model ----
set.seed(7)

# We apply a Leave One Out Cross Validation resampling method
train_control <- trainControl(method = "LOOCV")
# We also apply a standardize data transform to make the mean = 0 and
# standard deviation = 1
diabetes_caret_model_lda <- train(diabetes ~ .,
                                  data = pima_indians_diabetes_train,
                                  method = "lda", metric = "Accuracy",
                                  preProcess = c("center", "scale"),
                                  trControl = train_control)
### Display the model's details ----
print(diabetes_caret_model_lda)

### Make predictions ----
predictions <- predict(diabetes_caret_model_lda,
                       pima_indians_diabetes_test[, 1:8])

### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         pima_indians_diabetes_test[, 1:9]$diabetes)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 4. Regularized Regression ----
# The glmnet() function is in the glmnet package and can be used for classification or regression.
# It can also be configured to perform three important types of regularization: lasso, ridge and elastic net by configuring the alpha parameter to 1, 0 or in [0,1] respectively.
### 4.a. Classification Problem without CARET ----
# load the package
library(glmnet)
library(mlbench)
# load data
data(PimaIndiansDiabetes)
x <- as.matrix(PimaIndiansDiabetes[,1:8])
y <- as.matrix(PimaIndiansDiabetes[,9])
# fit model
fit <- glmnet(x, y, family="binomial", alpha=0.5, lambda=0.001)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x, type="class")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

### 4.b. Regression Problem without CARET ----
# caret version
# load the packages
library(glmnet)
library(mlbench)
# load data
data(BostonHousing)
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
x <- as.matrix(BostonHousing[,1:13])
y <- as.matrix(BostonHousing[,14])
# fit model
fit <- glmnet(x, y, family="gaussian", alpha=0.5, lambda=0.001)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x, type="link")
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)

### 4.c. Classification Problem with CARET ----
# load packages
library(caret)
library(mlbench)
library(glmnet)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.glmnet <- train(diabetes~., data=PimaIndiansDiabetes, method="glmnet",
                    metric="Accuracy", preProcess=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.glmnet)

### 4.d. Regression Problem with CARET ----
# load packages
library(caret)
library(mlbench)
library(glmnet)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.glmnet <- train(medv~., data=BostonHousing, method="glmnet", metric="RMSE",
                    preProcess=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.glmnet)

# B. Non-Linear Algorithms ----
## 1.  k-Nearest Neighbours ----
# The knn3() function is in the caret package and does not create a model. Instead it makes predictions from the training dataset directly. It can be used for classification or regression.

### 1.a. Classification Problem without CARET ----
# load the packages
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- knn3(diabetes~., data=PimaIndiansDiabetes, k=3)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8], type="class")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

### 1.b. Regression Problem without CARET ----
# load the packages
library(caret)
library(mlbench)
# load data
data(BostonHousing)
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
x <- as.matrix(BostonHousing[,1:13])
y <- as.matrix(BostonHousing[,14])
# fit model
fit <- knnreg(x, y, k=3)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x)
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)

### 1.c. Classification Problem with CARET ----
# load packages
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.knn <- train(diabetes~., data=PimaIndiansDiabetes, method="knn", metric="Accuracy",
                 preProcess=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.knn)

### 1.d. Regression Problem with CARET ----
# load packages
library(caret)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.knn <- train(medv~., data=BostonHousing, method="knn", metric="RMSE",
                 preProcess=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.knn)

## 2.  Naïve Bayes ----
### 2.a. Classification Problem without CARET ----
# load the packages
library(e1071)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- naiveBayes(diabetes~., data=PimaIndiansDiabetes)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8])
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

### 2.b. Classification Problem with CARET ----
# load packages
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.nb <- train(diabetes~., data=PimaIndiansDiabetes, method="nb", metric="Accuracy",
                trControl=trainControl)
# summarize fit
print(fit.nb)

## 3.  Support Vector Machine ----
# The ksvm() function is in the kernlab package and can be used for classification or regression.
### 3.a. Classification Problem without CARET ----
# The ksvm() function is in the kernlab package and can be used for classification or regression.
# load the packages
library(kernlab)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- ksvm(diabetes~., data=PimaIndiansDiabetes, kernel="rbfdot")
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8], type="response")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

### 3.b. Regression Problem without CARET ----
# load the packages
library(kernlab)
library(mlbench)
# load data
data(BostonHousing)
# fit model
fit <- ksvm(medv~., BostonHousing, kernel="rbfdot")
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, BostonHousing)
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)

### 3.c. Classification Problem with CARET ----
# The SVM with Radial Basis kernel implementation can be used with caret for classification as follows:
# load packages
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.svmRadial <- train(diabetes~., data=PimaIndiansDiabetes, method="svmRadial",
                       metric="Accuracy", trControl=trainControl)
# summarize fit
print(fit.svmRadial)

### 3.d. Regression Problem with CARET ----
# The SVM with Radial Basis kernel implementation can be used with caret for regression as follows:
# load packages
library(caret)
library(mlbench)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.svmRadial <- train(medv~., data=BostonHousing, method="svmRadial", metric="RMSE",
                       trControl=trainControl)
# summarize fit
print(fit.svmRadial)

## 4.  Classification and Regression Trees ----
### 4.a. Classification Problem without CARET ----
# load the packages
library(rpart)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- rpart(diabetes~., data=PimaIndiansDiabetes)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8], type="class")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

### 4.b. Regression Problem without CARET ----
# load the packages
library(rpart)
library(mlbench)
# load data
data(BostonHousing)
# fit model
fit <- rpart(medv~., data=BostonHousing, control=rpart.control(minsplit=5))
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, BostonHousing[,1:13])
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)

### 4.c. Classification Problem with CARET ----
# load packages
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.rpart <- train(diabetes~., data=PimaIndiansDiabetes, method="rpart", metric="Accuracy",
                   trControl=trainControl)
# summarize fit
print(fit.rpart)

### 4.d. Regression Problem with CARET ----
# load packages
library(caret)
library(mlbench)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=2)
fit.rpart <- train(medv~., data=BostonHousing, method="rpart", metric="RMSE",
                   trControl=trainControl)
# summarize fit
print(fit.rpart)

# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint

# References ----

## Kuhn, M., Wing, J., Weston, S., Williams, A., Keefer, C., Engelhardt, A., Cooper, T., Mayer, Z., Kenkel, B., R Core Team, Benesty, M., Lescarbeau, R., Ziem, A., Scrucca, L., Tang, Y., Candan, C., & Hunt, T. (2023). caret: Classification and Regression Training (6.0-94) [Computer software]. https://cran.r-project.org/package=caret # nolint ----

## Leisch, F., & Dimitriadou, E. (2023). mlbench: Machine Learning Benchmark Problems (2.1-3.1) [Computer software]. https://cran.r-project.org/web/packages/mlbench/index.html # nolint ----

## National Institute of Diabetes and Digestive and Kidney Diseases. (1999). Pima Indians Diabetes Dataset [Dataset]. UCI Machine Learning Repository. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database # nolint ----

## Robin, X., Turck, N., Hainard, A., Tiberti, N., Lisacek, F., Sanchez, J.-C., Müller, M., Siegert, S., Doering, M., & Billings, Z. (2023). pROC: Display and Analyze ROC Curves (1.18.4) [Computer software]. https://cran.r-project.org/web/packages/pROC/index.html # nolint ----

## Wickham, H., François, R., Henry, L., Müller, K., Vaughan, D., Software, P., & PBC. (2023). dplyr: A Grammar of Data Manipulation (1.1.3) [Computer software]. https://cran.r-project.org/package=dplyr # nolint ----

## Wickham, H., Chang, W., Henry, L., Pedersen, T. L., Takahashi, K., Wilke, C., Woo, K., Yutani, H., Dunnington, D., Posit, & PBC. (2023). ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics (3.4.3) [Computer software]. https://cran.r-project.org/package=ggplot2 # nolint ----

# **Required Lab Work Submission** ----
## Part A ----
# Create a new file called
# "Lab6-Submission-EvaluationMetrics.R".
# Provide all the code you have used to demonstrate the classification and
# regression evaluation metrics we have gone through in this lab.
# This should be done on any datasets of your choice except the ones used in
# this lab.

## Part B ----
# Upload *the link* to your
# "Lab6-Submission-EvaluationMetrics.R" hosted
# on Github (do not upload the .R file itself) through the submission link
# provided on eLearning.

## Part C ----
# Create a markdown file called "Lab-Submission-Markdown.Rmd"
# and place it inside the folder called "markdown". Use R Studio to ensure the
# .Rmd file is based on the "GitHub Document (Markdown)" template when it is
# being created.

# Refer to the following file in Lab 1 for an example of a .Rmd file based on
# the "GitHub Document (Markdown)" template:
#     https://github.com/course-files/BBT4206-R-Lab1of15-LoadingDatasets/blob/main/markdown/BIProject-Template.Rmd # nolint

# Include Line 1 to 14 of BIProject-Template.Rmd in your .Rmd file to make it
# displayable on GitHub when rendered into its .md version

# It should have code chunks that explain all the steps performed on the
# datasets.

## Part D ----
# Render the .Rmd (R markdown) file into its .md (markdown) version by using
# knitR in RStudio.

# You need to download and install "pandoc" to render the R markdown.
# Pandoc is a file converter that can be used to convert the following files:
#   https://pandoc.org/diagram.svgz?v=20230831075849

# Documentation:
#   https://pandoc.org/installing.html and
#   https://github.com/REditorSupport/vscode-R/wiki/R-Markdown

# By default, Rmd files are open as Markdown documents. To enable R Markdown
# features, you need to associate *.Rmd files with rmd language.
# Add an entry Item "*.Rmd" and Value "rmd" in the VS Code settings,
# "File Association" option.

# Documentation of knitR: https://www.rdocumentation.org/packages/knitr/

# Upload *the link* to "Lab-Submission-Markdown.md" (not .Rmd)
# markdown file hosted on Github (do not upload the .Rmd or .md markdown files)
# through the submission link provided on eLearning.