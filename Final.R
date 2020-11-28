install.packages('e1071')
install.packages('kknn')
install.packages("party")
install.packages("rsample")
install.packages('randomForest')
install.packages("gbm")
library(e1071)
library(kknn)
library(party)
library(rsample)
library(randomForest)
library(gbm)
data_train <- read.table('group2_trainset.csv',sep = ',', header = T)
data_test <- read.table('group2_testset.csv',sep = ',', header = T)

unique(data_train$veh_body)
data_train['veh_body_STNWG'] <- as.numeric(data_train$veh_body=='STNWG')
data_train['veh_body_HBACK'] <- as.numeric(data_train$veh_body=='HBACK')
data_train['veh_body_SEDAN'] <- as.numeric(data_train$veh_body=='SEDAN')
data_train['veh_body_UTE'] <- as.numeric(data_train$veh_body=='UTE')
data_train['veh_body_TRUCK'] <- as.numeric(data_train$veh_body=='TRUCK')
data_train['veh_body_HDTOP'] <- as.numeric(data_train$veh_body=='HDTOP')
data_train['veh_body_COUPE'] <- as.numeric(data_train$veh_body=='COUPE')
data_train['veh_body_PANVN'] <- as.numeric(data_train$veh_body=='PANVN')
data_train['veh_body_MIBUS'] <- as.numeric(data_train$veh_body=='MIBUS')
data_train['veh_body_MCARA'] <- as.numeric(data_train$veh_body=='MCARA')
data_train['veh_body_BUS'] <- as.numeric(data_train$veh_body=='BUS')
data_train['veh_body_CONVT'] <- as.numeric(data_train$veh_body=='CONVT')
data_train['veh_body_RDSTR'] <- as.numeric(data_train$veh_body=='RDSTR')
data_train['gender'] <- as.numeric(data_train$gender == 'M')
data_train['area_A'] <- as.numeric(data_train$area == 'A')
data_train['area_B'] <- as.numeric(data_train$area == 'B')
data_train['area_C'] <- as.numeric(data_train$area == 'C')
data_train['area_D'] <- as.numeric(data_train$area == 'D')
data_train['area_E'] <- as.numeric(data_train$area == 'E')
data_train['area_F'] <- as.numeric(data_train$area == 'F')

data_test['veh_body_STNWG'] <- as.numeric(data_test$veh_body=='STNWG')
data_test['veh_body_HBACK'] <- as.numeric(data_test$veh_body=='HBACK')
data_test['veh_body_SEDAN'] <- as.numeric(data_test$veh_body=='SEDAN')
data_test['veh_body_UTE'] <- as.numeric(data_test$veh_body=='UTE')
data_test['veh_body_TRUCK'] <- as.numeric(data_test$veh_body=='TRUCK')
data_test['veh_body_HDTOP'] <- as.numeric(data_test$veh_body=='HDTOP')
data_test['veh_body_COUPE'] <- as.numeric(data_test$veh_body=='COUPE')
data_test['veh_body_PANVN'] <- as.numeric(data_test$veh_body=='PANVN')
data_test['veh_body_MIBUS'] <- as.numeric(data_test$veh_body=='MIBUS')
data_test['veh_body_MCARA'] <- as.numeric(data_test$veh_body=='MCARA')
data_test['veh_body_BUS'] <- as.numeric(data_test$veh_body=='BUS')
data_test['veh_body_CONVT'] <- as.numeric(data_test$veh_body=='CONVT')
data_test['veh_body_RDSTR'] <- as.numeric(data_test$veh_body=='RDSTR')
data_test['gender'] <- as.numeric(data_test$gender == 'M')
data_test['area_A'] <- as.numeric(data_test$area == 'A')
data_test['area_B'] <- as.numeric(data_test$area == 'B')
data_test['area_C'] <- as.numeric(data_test$area == 'C')
data_test['area_D'] <- as.numeric(data_test$area == 'D')
data_test['area_E'] <- as.numeric(data_test$area == 'E')
data_test['area_F'] <- as.numeric(data_test$area == 'F')

set.seed(8051)

normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

formula1 = as.formula('claim_ind ~ veh_value + exposure  + veh_age + gender + dr_age + veh_body_STNWG + veh_body_HBACK + veh_body_SEDAN + veh_body_UTE + veh_body_TRUCK + veh_body_HDTOP + veh_body_COUPE+ veh_body_PANVN + veh_body_MIBUS + veh_body_MCARA + veh_body_BUS + veh_body_CONVT + veh_body_RDSTR + area_A + area_B + area_C + area_D + area_E + area_F')
formula2 = as.formula('claim_cost ~ veh_value + exposure  + veh_age + gender + dr_age + veh_body_STNWG + veh_body_HBACK + veh_body_SEDAN + veh_body_UTE + veh_body_TRUCK + veh_body_HDTOP + veh_body_COUPE+ veh_body_PANVN + veh_body_MIBUS + veh_body_MCARA + veh_body_BUS + veh_body_CONVT + veh_body_RDSTR + area_A + area_B + area_C + area_D + area_E + area_F')
formula3 = as.formula('claim_cost ~ (veh_value + exposure  + veh_age + gender + dr_age + veh_body_STNWG + veh_body_HBACK + veh_body_SEDAN + veh_body_UTE + veh_body_TRUCK + veh_body_HDTOP + veh_body_COUPE+ veh_body_PANVN + veh_body_MIBUS + veh_body_MCARA + veh_body_BUS + veh_body_CONVT + veh_body_RDSTR + area_A + area_B + area_C + area_D + area_E + area_F)^2')

Sensitivity <- matrix(nrow = 4, ncol = 4)

gridsearch <- function(par1, par2)
{
  s = 1
  for (i in par1)
  {
    t = 1
    for (j in par2)
    {
      m_SVM <- svm(formula1, data = data_train, type = 'C',gamma = i,cost = j)
      data_test$predout <- predict(m_SVM, newdata = data_test)
      Sensitivity[s,t] <- sum(data_test$predout == 1 & data_test$claim_ind == 1)/sum(data_test$claim_ind == 1)
      t <- t + 1
    }
    s <- s + 1
  }
  return(Sensitivity)
}
m_SVM <- svm(formula1, data = data_train, type = 'C',gamma = 100,cost = 10)
SVMPred <- predict(m_SVM)
sum(SVMPred==1)
data_test$ predout <- predict(m_SVM, newdata = data_test)
xtabs( ~ claim_ind + predout, data_test)

gammaList <- c(0.1, 1, 10, 100)
CList <- c(0.1, 1, 10, 100)
gridsearch(gammaList, CList)

set_split <- initial_split(data_train, prob = 0.75)
data_traintrain <- training(set_split)
data_traintest <- testing(set_split)

m_tree <-ctree(formula1, data = data_train)
TreePred <- predict(m_tree)
data_traintest$predprob <- predict(m_tree, newdata = data_traintest)

thresh <- seq(0,1,0.01)
SensitivityTree <- 0
for (j in seq(along=thresh))
{
  data_traintest$predout <- ifelse(data_traintest$predprob <= thresh[j], FALSE, TRUE)
  if ((sum(data_traintest$predout == 0 & data_traintest$claim_ind == 0)/sum(data_traintest$claim_ind == 0))>=0.5)
  {
    SensitivityTree <- sum(data_traintest$predout == 1 & data_traintest$claim_ind == 1)/sum(data_traintest$claim_ind == 1)
    threshTree <- j        
    break
  } 
}
data_test$predprobT <- predict(m_tree, newdata = data_test)
data_test$predoutT <- ifelse(data_test$predprobT <= 0.07, FALSE, TRUE)
sum(data_test$predoutT == 1 & data_test$claim_ind == 1)/sum(data_test$claim_ind == 1)


plot(1-Specificity,Sensitivity,type="l",xlim = c(0,1), ylim = c(0,1))
abline(0,1,lty=2)


KNNList <- c(1,3,5,10,20,50)
SensitivityKNN <- numeric(6)
threshKNN <- numeric(6)
gridsearchKNN <- function(k)
{
  t <- 1
  for (i in k)
  {
    m_KNN <- kknn(formula1, data_traintrain, data_traintest, k = i)
    data_traintest$predprob <- fitted(m_KNN)
    thresh <- seq(0,1,0.05)
    for (j in seq(along=thresh))
    {
      data_traintest$predout <- ifelse(fitted(m_KNN) <= thresh[j], FALSE, TRUE)
      if ((sum(data_traintest$predout == 0 & data_traintest$claim_ind == 0)/sum(data_traintest$claim_ind == 0))>=0.5)
      {
        SensitivityKNN[t] <- sum(data_traintest$predout == 1 & data_traintest$claim_ind == 1)/sum(data_traintest$claim_ind == 1)
        threshKNN[t] <- j        
        break
      } 
    }
    t <- t + 1
  }
  print(threshKNN)
  return(SensitivityKNN)
}

gridsearchKNN(KNNList)

m_KNN <- kknn(formula1, data_train, data_test, k = 10)
data_test$predoutK <- ifelse(fitted(m_KNN) <= 0, FALSE, TRUE)
sum(data_test$predoutK == 1 & data_test$claim_ind == 1)/sum(data_test$claim_ind == 1)

m_RF <- randomForest(formula1, data = data_traintrain)
data_traintest$predprobRF <- predict(m_RF, newdata = data_traintest)
data_traintest$predoutRF <- ifelse(data_traintest$predprobRF <= 0.056, FALSE, TRUE)
sum(data_traintest$predoutRF == 0 & data_traintest$claim_ind == 0)/sum(data_traintest$claim_ind == 0)
sum(data_traintest$predoutRF == 1 & data_traintest$claim_ind == 1)/sum(data_traintest$claim_ind == 1)
data_test$predprobRF <- predict(m_RF, newdata = data_test)
data_test$predoutRF <- ifelse(data_test$predprobRF <= 0.056, FALSE, TRUE)
sum(data_test$predoutRF == 0 & data_test$claim_ind == 0)/sum(data_test$claim_ind == 0)
sum(data_test$predoutRF == 1 & data_test$claim_ind == 1)/sum(data_test$claim_ind == 1)
#0.6604278

m_gl <- glm(formula1, data = data_traintrain)
data_traintest$predprobgl <- predict(m_gl, newdata = data_traintest)
data_traintest$predoutgl <- ifelse(data_traintest$predprobgl <= 0.066, FALSE, TRUE)
sum(data_traintest$predoutgl == 0 & data_traintest$claim_ind == 0)/sum(data_traintest$claim_ind == 0)
sum(data_traintest$predoutgl == 1 & data_traintest$claim_ind == 1)/sum(data_traintest$claim_ind == 1)
data_test$predprobgl <- predict(m_gl, newdata = data_test)
data_test$predoutgl <- ifelse(data_test$predprobgl <= 0.066, FALSE, TRUE)
sum(data_test$predoutgl == 0 & data_test$claim_ind == 0)/sum(data_test$claim_ind == 0)
sum(data_test$predoutgl == 1 & data_test$claim_ind == 1)/sum(data_test$claim_ind == 1)
#0.7299465

m_gbm <- gbm(formula1, data = data_traintrain, distribution = "bernoulli", n.trees = 1000, shrinkage = 0.01, cv.folds = 3)
data_traintest$predprobgbm <- predict(m_gbm, newdata = data_traintest)
data_traintest$predoutgbm <- ifelse(data_traintest$predprobgbm <= -2.7, FALSE, TRUE)
sum(data_traintest$predoutgbm == 0 & data_traintest$claim_ind == 0)/sum(data_traintest$claim_ind == 0)
sum(data_traintest$predoutgbm == 1 & data_traintest$claim_ind == 1)/sum(data_traintest$claim_ind == 1)
data_test$predprobgbm <- predict(m_gbm, newdata = data_test)
data_test$predoutgbm <- ifelse(data_test$predprobgbm <= -2.7, FALSE, TRUE)
sum(data_test$predoutgbm == 0 & data_test$claim_ind == 0)/sum(data_test$claim_ind == 0)
sum(data_test$predoutgbm == 1 & data_test$claim_ind == 1)/sum(data_test$claim_ind == 1)
#0.7379679
write.csv(data_test, file = "data_test_withP.csv")

OLS <- lm(formula2, data = data_train[data_train$claim_ind==1,])
data_test$predFinal <- 0
data_test$predFinal[data_test$predoutgbm==1] <- predict(OLS, newdata = data_test[data_test$predoutgbm==1,])

normalizedGini(data_test$claim_cost,data_test$predFinal)

hist(data_test$claim_cost[data_test$claim_ind ==1])

data_testFinal <- read.table('InsNova_test.csv',sep = ',', header = T)
data_testFinal['veh_body_STNWG'] <- as.numeric(data_testFinal$veh_body=='STNWG')
data_testFinal['veh_body_HBACK'] <- as.numeric(data_testFinal$veh_body=='HBACK')
data_testFinal['veh_body_SEDAN'] <- as.numeric(data_testFinal$veh_body=='SEDAN')
data_testFinal['veh_body_UTE'] <- as.numeric(data_testFinal$veh_body=='UTE')
data_testFinal['veh_body_TRUCK'] <- as.numeric(data_testFinal$veh_body=='TRUCK')
data_testFinal['veh_body_HDTOP'] <- as.numeric(data_testFinal$veh_body=='HDTOP')
data_testFinal['veh_body_COUPE'] <- as.numeric(data_testFinal$veh_body=='COUPE')
data_testFinal['veh_body_PANVN'] <- as.numeric(data_testFinal$veh_body=='PANVN')
data_testFinal['veh_body_MIBUS'] <- as.numeric(data_testFinal$veh_body=='MIBUS')
data_testFinal['veh_body_MCARA'] <- as.numeric(data_testFinal$veh_body=='MCARA')
data_testFinal['veh_body_BUS'] <- as.numeric(data_testFinal$veh_body=='BUS')
data_testFinal['veh_body_CONVT'] <- as.numeric(data_testFinal$veh_body=='CONVT')
data_testFinal['veh_body_RDSTR'] <- as.numeric(data_testFinal$veh_body=='RDSTR')
data_testFinal['gender'] <- as.numeric(data_testFinal$gender == 'M')
data_testFinal['area_A'] <- as.numeric(data_testFinal$area == 'A')
data_testFinal['area_B'] <- as.numeric(data_testFinal$area == 'B')
data_testFinal['area_C'] <- as.numeric(data_testFinal$area == 'C')
data_testFinal['area_D'] <- as.numeric(data_testFinal$area == 'D')
data_testFinal['area_E'] <- as.numeric(data_testFinal$area == 'E')
data_testFinal['area_F'] <- as.numeric(data_testFinal$area == 'F')

data_testFinal$predprobgbm <- predict(m_gbm, newdata = data_testFinal)
data_testFinal$predoutgbm <- ifelse(data_testFinal$predprobgbm <= -2.7, FALSE, TRUE)
data_testFinal$predFinal <- 0
data_testFinal$predFinal[data_testFinal$predoutgbm==1] <- predict(OLS, newdata = data_testFinal[data_testFinal$predoutgbm==1,])

data_testFinal2 <- read.table('InsNova_test.csv',sep = ',', header = T)
data_testFinal2$claim_cost <- data_testFinal$predFinal
write.csv(data_testFinal2, file = "data_test_withP2.csv")
