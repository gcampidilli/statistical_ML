# Data Mining and Machine Learning final project 
# Fall 2021
# Grace Campidilli (gec93), Ananya Jambhale (adj48), Medha Bulumulla (mb2569)

# Goal: Identify what method(s) are best at predicting wine quality
# Associated final report can be found at: https://docs.google.com/document/d/1LnQiNDo9rM8MYh0kvTxZ2vNjv9MkWV6ivgN795knP5M/edit?usp=sharing
# Data and reference paper can be found at: https://archive.ics.uci.edu/ml/datasets/wine+quality

# Outline
# PART 1: LOAD DATA AND REQUIRED PACKAGES
# PART 2: LINEAR MODEL SELECTION - linear regression, best subset selection, polynomial regression, shrinkage methods (ridge and lasso)
# PART 3: NON-LINEAR MODEL SELECTION - smoothing splines, local regression, general additive models, dimension reductiom (PCA and PLS)
# PART 4: TREE METHODS - general tree regression, bagging method, random forest
# PART 5: CROSS VALIDATION FOR BEST MODEL FROM EACH CATEGORY


#################################################################################
# PART 1: LOAD DATA AND REQUIRED PACKAGES

library(ISLR)
library(plyr)
library(Metrics)

wd = "/Users/gcampidilli/Documents/ML"
setwd(wd)
wine_dat = read.csv("wine-quality-white-and-red.csv", header = T)
# Input variables (based on physicochemical tests):
# 1 - fixed acidity
# 2 - volatile acidity
# 3 - citric acid
# 4 - residual sugar
# 5 - chlorides
# 6 - free sulfur dioxide
# 7 - total sulfur dioxide
# 8 - density
# 9 - pH
# 10 - sulphates
# 11 - alcohol
# Output variable (based on sensory data):
#   12 - quality (score between 0 and 10)

# data summary and basic exploration

# Initializing train and test dataset 
set.seed(1)
train = sample(1:nrow(wine_dat), nrow(wine_dat)/2)
test_data = wine_dat[-train,]
wine.quality.test = test_data[,"quality"]

# default boot func was not working with updated R version, so we made our own
boot_mse_func = function(y.data){
  save = vector()
  for (i in 1:1000){
    index = sample(1:length(y.data), replace = TRUE)
    save = c(save, mse(wine.quality.test[index],y.data[index]))
    return(save)
  }
}

#################################################################################
# PART 2: LINEAR MODEL SELECTION

# Least squares regression

least_squares_fit = lm(quality ~., data = wine_dat, subset = train)
yhat = round_any(predict(least_squares_fit, newdata = test_data),1)
leastsquares.mse = round(mean(boot_mse_func(yhat)), digits = 5)

print(paste("Least Squares model has MSE of", leastsquares.mse))

# Reduced least squares regression
summary(least_squares_fit) # drop predictors that don't have pvalues ***

reduced_least_squares = lm(quality ~ .-citric.acid - chlorides, data = wine_dat, subset = train)

anova(least_squares_fit, reduced_least_squares)

yhat.ls2 = round_any(predict(reduced_least_squares, newdata = test_data),1)
reducedls.reduced.mse = round(mean(boot_mse_func(yhat.ls2)), digits = 5)

print(paste("Reduced Least Squares model has MSE of", reducedls.reduced.mse))

# Best subset selection
library(leaps)
bestRegFit = regsubsets(quality ~ . , subset = train, data = wine_dat)
print(summary(bestRegFit))

# best 1 predictor model - Alcohol
best.predictor.1 = lm(quality~alcohol, subset = train, data = wine_dat)
yhat.best1 = round_any(predict(best.predictor.1, newdata = test_data),1)
best1.mse = round(mean(boot_mse_func(yhat.best1)), digits = 5)
print(paste("Best-Subset selected 1-predictor model has MSE of", best1.mse))

# best 2 predictor model - Alcohol + volatile.acidity
best.predictor.2 = lm(quality~alcohol + volatile.acidity, subset = train, data = wine_dat)
yhat.best2 = round_any(predict(best.predictor.2, newdata = test_data),1)
best2.mse = round(mean(boot_mse_func(yhat.best2)), digits = 5)
print(paste("Best-Subset selected 2-predictor model has MSE of", best2.mse))

# best 3 predictor model - Alcohol + volatile.acidity + sulphates
best.predictor.3 = lm(quality~alcohol + volatile.acidity + sulphates, subset = train, data = wine_dat)
yhat.best3 = round_any(predict(best.predictor.3, newdata = test_data),1)
best3.mse = round(mean(boot_mse_func(yhat.best3)), digits = 5)
print(paste("Best-Subset selected 3-predictor model has MSE of", best3.mse))

# best 4 predictor model - Alcohol + volatile.acidity + sulphates + residual.sugar
best.predictor.4 = lm(quality~alcohol + volatile.acidity + sulphates + residual.sugar, subset = train, data = wine_dat)
yhat.best4 = round_any(predict(best.predictor.4, newdata = test_data),1)
best4.mse = round(mean(boot_mse_func(yhat.best4)), digits = 5)
print(paste("Best-Subset selected 4-predictor model has MSE of", best4.mse))

# best 5 predictor model - Alcohol + volatile.acidity + sulphates + residual.sugar + free.sulfur.dioxide
best.predictor.5 = lm(quality~alcohol + volatile.acidity + sulphates + residual.sugar + free.sulfur.dioxide, subset = train, data = wine_dat)
yhat.best5 = round_any(predict(best.predictor.5, newdata = test_data),1)
best5.mse = round(mean(boot_mse_func(yhat.best5)), digits = 5)
print(paste("Best-Subset selected 5-predictor model has MSE of", best5.mse))

set.seed(1)
# Polynomial Regression
pwr = 2
poly.fit = lm(quality~ poly(alcohol,pwr) + poly(volatile.acidity,pwr) + poly(sulphates,pwr) +
              poly(residual.sugar,pwr) + poly(free.sulfur.dioxide,pwr), data = wine_dat, subset = train)
summary(poly.fit)

# sulphates pwr 2 is not significant, residual.sugar pwr 2 is not significant
pwr = 3
poly.fit.2 =  lm(quality~ poly(alcohol,pwr) + poly(volatile.acidity,pwr) + sulphates +
                   residual.sugar + poly(free.sulfur.dioxide,pwr), data = wine_dat, subset = train)
summary(poly.fit.2)

# free.sulfur.dioxide pwr 3 is not significant
pwr = 3
poly.fit.3 = lm(quality~ poly(alcohol,pwr) + poly(volatile.acidity,pwr) + sulphates +
                  residual.sugar + poly(free.sulfur.dioxide,pwr), data = wine_dat, subset = train)
summary(poly.fit.3)

pwr = 4
poly.fit.4 = lm(quality~ poly(alcohol,pwr) + poly(volatile.acidity,pwr) + sulphates +
                  residual.sugar + poly(free.sulfur.dioxide,2), data = wine_dat, subset = train)
summary(poly.fit.4)

# alcohol pwr 4 is not significant
poly.fit.final = lm(quality~ poly(alcohol,3) + poly(volatile.acidity,4) + sulphates +
                      residual.sugar + poly(free.sulfur.dioxide,2), data = wine_dat, subset = train)

anova(poly.fit.4, poly.fit.final)

yhat.poly = round_any(predict(poly.fit.final, newdata = test_data),1)
poly.mse = round(mean(boot_mse_func(yhat.poly)), digits = 5)
print(paste("Best-Subset selected 5-predictor model has MSE of", poly.mse))


# Ridge regression
x = model.matrix(quality~.,wine_dat)[train,-1]
y = wine_dat$quality[train]
x.test = model.matrix(quality~.,wine_dat)[-train,-1]

library(glmnet)
# choose value for lambda
ridge.cv = cv.glmnet(x,y, alpha =0)
bestlam = ridge.cv$lambda.min
ridge.wine=glmnet(x,y,alpha=0,lambda=bestlam, thresh =1e-12)

yhat.ridge=predict(ridge.wine,s=bestlam ,newx= x.test)

ridge.mse = round(mean(boot_mse_func(yhat.ridge)), digits = 5)
print(paste("Ridge regression has MSE of", ridge.mse))

# Lasso
library(glmnet)

x = model.matrix(quality~.,wine_dat)[train,-1]
y = wine_dat$quality[train]
x.test = model.matrix(quality~.,wine_dat)[-train,-1]

grid=10^seq(10,-2,length=100)
lasso.mod=glmnet(x,y,alpha=1,lambda=grid)

cv.out=cv.glmnet(x,y,alpha=1)
bestlam=cv.out$lambda.min
yhat.lasso=predict(lasso.mod,s=bestlam ,newx=x.test)
lasso.wine.mse = round(mean(boot_mse_func(yhat.lasso)), digits = 5)
print(paste("Lasso has MSE of", lasso.wine.mse))

# feature selection 
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:13,]
lasso.coef # determine which predictors were assigned 0

#################################################################################
# PART 3: NON-LINEAR MODEL SELECTION

# smoothing splines
library(splines)
train_dat = wine_dat[train,]
fit=smooth.spline(train_dat$quality ~ train_dat$alcohol+train_dat$fixed.acidity+train_dat$volatile.acidity+train_dat$citric.acid+
                    train_dat$residual.sugar+train_dat$chlorides+train_dat$free.sulfur.dioxide+train_dat$total.sulfur.dioxide+train_dat$density+train_dat$pH+train_dat$sulphates)
summary(fit)
yhat.smoothingspline =predict(fit, newdata=test)$y
smoothingspline.mse = round(mean(boot_mse_func(yhat.smoothingspline)), digits = 5)
print(paste("Smoothing spline has MSE of", smoothingspline.mse))

# local regression - can only choose 4 predictors
library(splines)
local.fit=loess(train_dat$quality ~ train_dat$volatile.acidity+train_dat$residual.sugar+train_dat$sulphates+train_dat$alcohol)
yhat.local = round_any(predict(local.fit, newdata = test_data[-c(1),]),1)
localreg.mse = round(mean(boot_mse_func(yhat.local)), digits = 5)
print(paste("Local regression has MSE of", localreg.mse))

# General additive models
library(gam)
gam.fit = gam(quality ~ s(alcohol) + s(chlorides) + s(fixed.acidity) + s(volatile.acidity) +s(residual.sugar) + s(sulphates) + s(citric.acid)+
              + s(free.sulfur.dioxide) + s(total.sulfur.dioxide)+ s(density) + s(pH) + s(sulphates) +type, data=train_dat)
yhat.gam = round_any(predict(gam.fit, newdata=test_data), 1)
gam.mse = round(mean(boot_mse_func(yhat.gam)), digits = 5)
print(paste("General additive model has MSE of", gam.mse))

# Principal Components Regression - principle component regression
library(pls)
pcr.fit=pcr(quality~., data=wine_dat, subset=train, scale=TRUE,validation="CV")
summary(pcr.fit)  # M=12 has smallest CV error, but not much different than M = 10
pcr.pred=predict(pcr.fit, newdata=wine_dat[-train,], ncomp=10) 
pcr.mse = round(mean(boot_mse_func(pcr.pred)), digits = 5)
print(paste("Principle component regression has MSE of", pcr.mse))


# partial least squares
set.seed(1)
pls.fit= plsr(quality~., data=wine_dat, subset=train, scale=TRUE,validation="CV")
summary(pls.fit)
pls.pred=predict(pls.fit, newdata=wine_dat[-train,], ncomp=8)
pls.mse = round(mean(boot_mse_func(pls.pred)), digits = 5)
print(paste("Partial least squares has MSE of", pls.mse))

#################################################################################
# PART 4: TREE METHODS

library(tree)
# tree regression
tree.wine = tree(quality~.,wine_dat, subset=train)
summary(tree.wine)
plot(tree.wine)
text(tree.wine, pretty = 0)

# use cross validation to determine how many terminal nodes produces the smallest error
cv.wine = cv.tree(tree.wine)
min.idx = which.min(cv.wine$dev)
best.size = cv.wine$size[min.idx]
prune.wine <- prune.tree(tree.wine, best = best.size)

yhat.treereg = round_any(predict(prune.wine, newdata=test_data), 1)
tree.mse = round(mean(boot_mse_func(yhat.treereg)), digits = 5)

print(paste("Tree regression has MSE of", tree.mse))

# Bagging
library(randomForest)
p = ncol(wine_dat)-1
bag.wine = randomForest(quality~., data=wine_dat, subset = train, mtry=p, importance=T)
yhat.bag = round_any(predict(bag.wine, newdata=test_data), 1)
bag.mse =round(mean(boot_mse_func(yhat.bag)), digits = 5)

print(paste("Bagging Tree has MSE of", bag.mse))

# Random Forest 
rf.wine = randomForest(quality~.,data=wine_dat, subset=train, importance=T)
yhat.rf = round_any(predict(rf.wine, newdata=test_data), 1)
rf.mse = round(mean(boot_mse_func(yhat.rf)), digits = 5)

print(paste("RandomForest Tree has MSE of", rf.mse))

plot(rf.wine, main = "# of Random Forest Trees vs Error")

#################################################################################
# PART 5: CROSS VALIDATION FOR BEST MODEL FROM EACH CATEGORY

library(tree)

# calculate LOOCV for Random Forest Tree
cv.rf = rfcv(wine_dat[train,], trainy = wine_dat[train,"quality"], cv.fold = nrow(wine_dat))
print(paste("Random Forest Tree has LOOCV error of", cv.rf$error.cv[1]))

# calculate LOOCV for Ridge Regression
library(glmnet)
x = model.matrix(quality~.,wine_dat)[train,-1]
y = wine_dat$quality[train]
x.test = model.matrix(quality~.,wine_dat)[-train,-1]

ridge.cv = cv.glmnet(x,y, alpha =0, nfolds = 10)
ridge.error = ridge.cv$cvm[which(ridge.cv$lambda== ridge.cv$lambda.min)]
print(paste("Ridge regression has LOOCV error of", ridge.error))

# summary figures for Random Forest
test_mse_vector = vector()
percent_error_vector = vector()

for(x in 1:50){
  set.seed(x)
  rf.wine = randomForest(quality~.,data=wine_dat, subset=train, importance=T)
  yhat.rf = round_any(predict(rf.wine, newdata=test_data), 1)
  rf.mse = round(mean(boot_mse_func(yhat.rf)), digits = 5)
  test_mse_vector[x] = rf.mse
  percent_error_vector[x] = mean(yhat.rf != wine.quality.test)
}

hist(test_mse_vector, breaks = 8, main = "Distribution of Test MSE for Random Forest Model", xlab = "test mse")
hist(percent_error_vector, breaks = 6, main = "Distribution of Percent error for Random Forest Model", xlab = "percent error")


