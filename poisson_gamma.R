library(data.table)
library(tidymodels)
library(ggplot2)
library(forcats)
library(MLmetrics)
library(pscl)

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

cutoff <- mean(train_full$claim_ind)

# step 1 - poisson -----
x <- train_set %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_count)
pois <- hurdle(claim_count ~ ., 
                 data = x)
summary(pois)

newx <- eval_set %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_count)
y_pred <- predict(pois, newdata = newx, type = "response")
hist(y_pred)
AUC(y_pred, y_true = eval_set$claim_ind)
Gini(y_pred = y_pred, y_true = eval_set$claim_ind)
Sensitivity(y_true = eval_set$claim_ind, y_pred = 1*(y_pred>=cutoff))
table(1*(y_pred>=cutoff), eval_set$claim_ind) 

# step 2 - gamma -----
x_train <- train_set %>% 
  filter(claim_ind == 1) %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_cost)
gm <- glm(claim_cost ~ ., 
          data = x_train, 
          family = Gamma(link = "log"))
newx2 <- eval_set[y_pred >= cutoff] %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_cost, id)
y_pred2 <- predict(gm, newdata = newx2, type = "response")
Gini(y_pred = y_pred2, y_true = newx2$claim_cost)

y_eval <- bind_rows(
  data.frame(id = eval_set[y_pred < cutoff, id], y_pred = 0, y_true = eval_set[y_pred < cutoff, claim_cost]),
  data.frame(id = newx2$id, y_pred = y_pred2, y_true = newx2$claim_cost)
)
Gini(y_eval$y_pred, y_eval$y_true)

# refit with full train data ----
x <- train_full %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_count)
pois_final <- update(pois, data = x)
newx <- test_set %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age)
y_pred <- predict(pois_final, newdata = newx, type = "response")
y_output <- data.frame(id = test_set$id, claim_cost = 0)[y_pred < cutoff,]

x2 <- train_full %>% 
  filter(claim_ind == 1) %>% 
  mutate(veh_value = scale(veh_value), 
         exposure_sq = exposure^2,
         veh_value_sq = scale(veh_value_sq)) %>% 
  select(exposure, exposure_sq, veh_value, veh_value_sq, veh_body, veh_age, 
         gender, area, dr_age, claim_cost)
gm_final <- update(gm, data = x2)
newx <- test_set[y_pred >= cutoff,] %>% 
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
         mutate(id = 1:nrow(y_output)), "hurdle_poisson_gamma_predictions.csv")
