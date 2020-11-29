library(data.table)
library(tidymodels)
library(ggplot2)
library(forcats)
library(MLmetrics)

train_full <- fread("InsNova_train.csv")
train_full[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2, 
  veh_body = fct_lump_prop(veh_body, prop = 0.05),
  veh_age = factor(veh_age), 
  dr_age = factor(dr_age)
)]
set.seed(8051)
set_split <- initial_split(train_full, prob = 0.75, strata = "claim_ind")

train_set <- training(set_split)
eval_set <- testing(set_split)

test_set <- fread("InsNova_test.csv")
test_set[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2, 
  veh_body = fct_lump_prop(veh_body, prop = 0.05),
  veh_age = factor(veh_age), 
  dr_age = factor(dr_age)
)]

# step 1 - logistic -----
library(glmnet)

x <- model.matrix(claim_ind ~ . - 1, 
                  data = train_set %>% 
                    mutate(veh_value = scale(veh_value), 
                           exposure_sq = exposure^2,
                           veh_value_sq = scale(veh_value_sq)) %>% 
                    select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
                           gender, area, dr_age, claim_ind))
lr_cv <- cv.glmnet(x = x, 
                y = train_set$claim_ind, 
                family = "binomial", 
                alpha = 0, 
                weights = ifelse(train_set$claim_ind == 1, 15, 1), 
                type.measure = "auc")
plot(lr_cv)
lr <- glmnet(x, y = train_set$claim_ind, 
             family = "binomial", 
             alpha = 0, 
             weights = ifelse(train_set$claim_ind == 1, 15, 1),
             lambda = lr_cv$lambda.min)
y_pred <- predict(lr, newx = model.matrix(claim_ind ~ . - 1, 
                                          data = eval_set %>% 
                                            mutate(veh_value = scale(veh_value), 
                                                   exposure_sq = exposure^2,
                                                   veh_value_sq = scale(veh_value_sq)) %>% 
                                            select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
                                                   gender, area, dr_age, claim_ind)), 
                  type = "response")
hist(y_pred)
AUC(y_pred, y_true = eval_set$claim_ind)
Gini(y_pred = y_pred, y_true = eval_set$claim_ind)
Sensitivity(y_true = eval_set$claim_ind, y_pred = 1*(y_pred>=.5))
table(1*(y_pred>=.5), eval_set$claim_ind) 

# step 2 - gamma -----
x_train <- train_set %>% 
  filter(claim_ind == 1) %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_cost)
gm <- glm(claim_cost ~ . + offset(exposure), 
          data = x_train, 
          family = Gamma(link = "log"))
newx2 <- eval_set[y_pred[,1] >= .5] %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_cost, id)
y_pred2 <- predict(gm, newdata = newx2, type = "response")
Gini(y_pred = y_pred2, y_true = newx2$claim_cost)

y_eval <- bind_rows(
  data.frame(id = eval_set[y_pred[,1] < .5, id], y_pred = 0, y_true = eval_set[y_pred[,1] < .5, claim_cost]),
  data.frame(id = newx2$id, y_pred = y_pred2, y_true = newx2$claim_cost)
)
Gini(y_eval$y_pred, y_eval$y_true)

# refit with full train data ----
x <- model.matrix(claim_ind ~ . - 1, 
                  data = train_full %>% 
                    mutate(veh_value = scale(veh_value), 
                           exposure_sq = exposure^2,
                           veh_value_sq = scale(veh_value_sq)) %>% 
                    select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
                           gender, area, dr_age, claim_ind))
lr_final <- glmnet(x = x, y = train_full$claim_ind, 
                   family = "binomial", 
                   alpha = 0, 
                   weights = ifelse(train_full$claim_ind == 1, 15, 1),
                   lambda = lr_cv$lambda.min)
newx <- model.matrix(claim_ind ~ . - 1, 
                     data = test_set %>% 
                       mutate(veh_value = scale(veh_value), 
                              exposure_sq = exposure^2,
                              veh_value_sq = scale(veh_value_sq), 
                              claim_ind = 0) %>% 
                       select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
                              gender, area, dr_age, claim_ind))
y_pred <- predict(lr_final, newx = newx, type = "response")
y_output <- data.frame(id = test_set$id, claim_cost = 0)[y_pred[,1] < .5,]

x2 <- train_full %>% 
  filter(claim_ind == 1) %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_cost)
gm_final <- glm(claim_cost ~ . + offset(exposure),
                data = x2, 
                family = Gamma(link = "log"))
newx <- test_set[y_pred[,1] >= .5,] %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, id)
y_pred2 <- predict(gm_final, newdata = newx, type = "response")
y_output <- bind_rows(
  y_output, 
  data.frame(id = newx$id, claim_cost = y_pred2)
)

fwrite(y_output %>% 
         arrange(id) %>% 
         mutate(id = 1:nrow(y_output)), "logistic_gamma_predictions.csv")
