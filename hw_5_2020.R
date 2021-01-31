setwd('../iamchetry/Documents/UB_files/506/hw_5/')
#install.packages("randomForest")
#install.packages("rpart")
#install.packages("bootstrap")

library(rpart) 
library(MASS)
library(caret)
library(dplyr)
library(glue)
library(leaps)
library(pROC)
library(randomForest)
library(bootstrap)


#--------------- 1st Question -----------------

load('vehicle.RData')
data_ = vehicle[, -19]
attach(data_)
set.seed(1)
control_ <- rpart.control(minsplit = 20, xval = 5, cp = 0)
tree_ <- rpart(class~., data = data_, method = "class", control = control_)

plot(tree_$cptable[,4], main = "Cp for model selection", ylab = "Cp")

min_cp = which.min(tree_$cptable[,4])
pruned_tree = prune(tree_, cp = tree_$cptable[min_cp,1])

#Feature Importance
plot(pruned_tree$variable.importance, xlab="variable", 
     ylab="Importance", xaxt = "n", pch=20)
axis(1, at=1:length(pruned_tree$variable.importance), 
     labels=names(pruned_tree$variable.importance))

par(mfrow = c(1,2))
plot(pruned_tree, branch = .3, compress=T, main = "Pruned Tree")
text(pruned_tree, cex = .5)

plot(tree_, branch = .3, compress=T, main = "Full Tree")
text(tree_, cex = .5)


#--------------- 2nd Question -----------------

data_ = data.frame(read.table('prostate.data.txt'))
data_$train = as.numeric(data_$train)
attach(data_)

set.seed(2)
t = createDataPartition(train, p=0.7, list = FALSE)
train_ = na.omit(data_[t, ])
test_ = na.omit(data_[-t, ])

fit = regsubsets(train~., data = train_, nbest = 1, nvmax = 9, 
                 method = "exhaustive")
mysum = summary(fit)

par(mfrow = c(2, 1))
plot(1:9, mysum$cp, type='line', main='Cp Values')
plot(1:9, mysum$bic, type='line', main='BIC Values')

#Bootstrap Error----------------------
beta.fit = function(X,Y){
  lsfit(X,Y)	
}

beta.predict = function(fit, X){
  cbind(1,X)%*%fit$coef
}

sq.error = function(Y,Yhat){
  abs(Y-Yhat)
}

# Create X and Y
X = data_[, 1:9]
Y = data_[, 10]

select = summary(fit)$outmat

boot_errors = c()
for (i in 1:9){
  temp = which(select[i,] == "*")
  
  res = bootpred(X[, temp], Y, nboot = 50, theta.fit = beta.fit,
                 theta.predict = beta.predict, err.meas = sq.error) 
  boot_errors = c(boot_errors, round(res[[3]], 4))
  
}

#5 Fold CV-------------------
set.seed(21)
df = split(data_, sample(1:5, nrow(data_), replace=T))

appended_errors = list()
appended_train = list()

for (k in 1:5)
{
  train_ = data_[!rownames(data_) %in% rownames(as.data.frame(df[glue('{k}')])), ]
  test_ = data.frame(df[glue('{k}')])
  names(test_) = names(train_)

  temp_train = cbind(rep(1, length(train_[, 1])), train_) # Creating Intercept Column
  names(temp_train) = c('(Intercept)', names(train_))
  temp_test = cbind(rep(1, length(test_[, 1])), test_)
  names(temp_test) = c('(Intercept)', names(test_)) 
  
  y_train = as.numeric(train_$train)
  y_test = as.numeric(test_$train)
  
  best_sub = regsubsets(train~., data = train_, nbest = 1, nvmax = 9,
                        method = "exhaustive")
  best_summary = summary(best_sub)
  
  best_errors = c()
  train_errors = c()
  
  for (i in 1:9)
  {
    coeff_ = coef(best_sub, id=i)
    d = temp_test[c(names(coeff_))]
    test_preds = t(coeff_%*%t(d))
    
    d = temp_train[c(names(coeff_))]
    train_preds = t(coeff_%*%t(d))
    
    #Finding optimal threshold to calculate decision boundary
    analysis = roc(response=train_$train, predictor=train_preds)
    e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
    opt_t = subset(e,e[,2]==max(e[,2]))[,1]

    test_preds = ifelse(test_preds >= opt_t, 1, 0) # Test Prediction
    test_error = round(sum(abs(y_test- test_preds))/length(y_test), 4) # Testing Error
    best_errors[i] = test_error
    
    train_preds = ifelse(train_preds >= opt_t, 1, 0) # Train Prediction
    train_error = round(sum(abs(y_train- train_preds))/length(y_train), 4) # Train Error
    train_errors[i] = train_error
  }
  appended_errors = c(appended_errors, best_errors)
  appended_train = c(appended_train, train_errors)
}

# Train Errors
train_errors_five_fold = c()
for (i in 1:9)
{
  l = c()
  for (k in seq(from=i, to=45, by=9))
  {
    l = c(l, appended_train[[k]])
  }
  train_errors_five_fold = c(train_errors_five_fold, mean(l))
}

# Test Errors
pred_errors_five_fold = c()
for (i in 1:9)
{
  l = c()
  for (k in seq(from=i, to=45, by=9))
  {
    l = c(l, appended_errors[[k]])
  }
  pred_errors_five_fold = c(pred_errors_five_fold, mean(l))
}

x = 1:9
y1 = boot_errors
y2 = train_errors_five_fold
y3 = pred_errors_five_fold
df = data.frame(x, y1, y2, y3)

# Combined plot of different Error types
require(ggplot2)
ggplot(df, aes(x)) +                    
  geom_line(aes(y=y1), colour="red") + 
  geom_line(aes(y=y2), colour="green") +
  geom_line(aes(y=y3), colour="blue")


#10 Fold CV-------------------
set.seed(22)
df = split(data_, sample(1:10, nrow(data_), replace=T))

appended_errors = list()
appended_train = list()

for (k in 1:10)
{
  train_ = data_[!rownames(data_) %in% rownames(as.data.frame(df[glue('{k}')])), ]
  test_ = data.frame(df[glue('{k}')])
  names(test_) = names(train_)
  
  temp_train = cbind(rep(1, length(train_[, 1])), train_) # Creating Intercept Column
  names(temp_train) = c('(Intercept)', names(train_))
  temp_test = cbind(rep(1, length(test_[, 1])), test_)
  names(temp_test) = c('(Intercept)', names(test_)) 
  
  y_train = as.numeric(train_$train)
  y_test = as.numeric(test_$train)
  
  best_sub = regsubsets(train~., data = train_, nbest = 1, nvmax = 9,
                        method = "exhaustive")
  best_summary = summary(best_sub)
  
  best_errors = c()
  train_errors = c()
  
  for (i in 1:9)
  {
    coeff_ = coef(best_sub, id=i)
    d = temp_test[c(names(coeff_))]
    test_preds = t(coeff_%*%t(d))
    
    d = temp_train[c(names(coeff_))]
    train_preds = t(coeff_%*%t(d))
    
    #Finding optimal threshold to calculate decision boundary
    analysis = roc(response=train_$train, predictor=train_preds)
    e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
    opt_t = subset(e,e[,2]==max(e[,2]))[,1]

    test_preds = ifelse(test_preds >= opt_t, 1, 0) # Test Prediction
    test_error = round(sum(abs(y_test- test_preds))/length(y_test), 4) # Testing Error
    best_errors[i] = test_error
    
    train_preds = ifelse(train_preds >= opt_t, 1, 0) # Train Prediction
    train_error = round(sum(abs(y_train- train_preds))/length(y_train), 4) # Train Error
    train_errors[i] = train_error
  }
  appended_errors = c(appended_errors, best_errors)
  appended_train = c(appended_train, train_errors)
}

# Train Errors
train_errors_ten_fold = c()
for (i in 1:9)
{
  l = c()
  for (k in seq(from=i, to=90, by=9))
  {
    l = c(l, appended_train[[k]])
  }
  train_errors_ten_fold = c(train_errors_ten_fold, mean(l))
}

# Test Errors
pred_errors_ten_fold = c()
for (i in 1:9)
{
  l = c()
  for (k in seq(from=i, to=90, by=9))
  {
    l = c(l, appended_errors[[k]])
  }
  pred_errors_ten_fold = c(pred_errors_ten_fold, mean(l))
}

x = 1:9
y1 = boot_errors
y2 = train_errors_ten_fold
y3 = pred_errors_ten_fold
df = data.frame(x, y1, y2, y3)

# Combined plot of different Error types
require(ggplot2)
ggplot(df, aes(x)) +                    
  geom_line(aes(y=y1), colour="red") + 
  geom_line(aes(y=y2), colour="green") +
  geom_line(aes(y=y3), colour="blue")


#------------------ 3rd Question ---------------------

data_wine = data.frame(read.table('wine.data', sep = ','))
data_wine$V1 = as.factor(data_wine$V1)
attach(data_wine)
set.seed(3)
t = createDataPartition(V1, p=0.7, list = FALSE)
train_ = na.omit(data_wine[t, ])
test_ = na.omit(data_wine[-t, ])
y_true = test_$V1

#Single Tree Prediction
control_ <- rpart.control(minsplit = 10, xval = 5, cp = 0)
tree_ <- rpart(V1~., data = train_, method = "class", control = control_)

plot(tree_$cptable[,4], main = "Cp for model selection", ylab = "Cp")

min_cp = which.min(tree_$cptable[,4])
pruned_tree <- prune(tree_, cp = tree_$cptable[min_cp,1])

#Feature Importance
plot(pruned_tree$variable.importance, xlab="variable", 
     ylab="Importance", xaxt = "n", pch=20)
axis(1, at=1:length(pruned_tree$variable.importance), 
     labels=names(pruned_tree$variable.importance))

par(mfrow = c(1,2))
plot(pruned_tree, branch = .3, compress=T, main = "Pruned Tree")
text(pruned_tree, cex = .5)

plot(tree_, branch = .3, compress=T, main = "Full Tree")
text(tree_, cex = .5)

my_pred = predict(pruned_tree, newdata = test_, type = "class")
y_hat = as.numeric(my_pred)

tab_test = table(my_pred, y_true)
conf_test = confusionMatrix(tab_test)

test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


#Bagging
rf_model <- randomForest(V1~., data = train_, ntree = 10000, mtry = 10)
par(mfrow = c(1,2))
varImpPlot(rf_model, main='Feature Importances')
importance(rf_model)

rf_pred = predict(rf_model, newdata = test_, type = "response")
tab_test = table(rf_pred, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


#LDA
lda_ = lda(V1~., data=train_)

lda_preds = predict(lda_, newdata = test_)
tab_test = table(lda_preds$class, y_true)

#--------- Confusion Matrix to determine Accuracy ---------
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


#----------------- 4th Question ------------------

load('covertype.RData')
covertype$V55 = as.factor(covertype$V55)
attach(covertype)
set.seed(4)
t = createDataPartition(V55, p=0.7, list = FALSE)
train_ = na.omit(covertype[t, ])
test_ = na.omit(covertype[-t, ])

train_true = train_$V55
test_true = test_$V55

#Single Tree Prediction
control_ = rpart.control(minsplit = 50, xval = 5, cp = 0)
tree_ = rpart(V55~., data = train_, method = "class", control = control_)

plot(tree_$cptable[,4], main = "Cp for model selection", ylab = "Cp")

min_cp = which.min(tree_$cptable[,4])
pruned_tree = prune(tree_, cp = tree_$cptable[min_cp,1])

#Feature Importance
plot(pruned_tree$variable.importance, xlab="variable", 
     ylab="Importance", xaxt = "n", pch=20)
axis(1, at=1:length(pruned_tree$variable.importance),
     labels=names(pruned_tree$variable.importance))

par(mfrow = c(1,2))
plot(pruned_tree, branch = .3, compress=T, main = "Pruned Tree")
text(pruned_tree, cex = .5)

plot(tree_, branch = .3, compress=T, main = "Full Tree")
text(tree_, cex = .5)

pred_train = predict(pruned_tree, newdata = train_, type = "class")
tab_train = table(pred_train, train_true)
conf_train = confusionMatrix(tab_train)
train_error = 1 - round(conf_train$overall['Accuracy'], 4) # Training Error

pred_test = predict(pruned_tree, newdata = test_, type = "class")
tab_test = table(pred_test, test_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


