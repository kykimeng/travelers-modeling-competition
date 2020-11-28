library(cplm)
library(GlmSimulatoR)
library(data.table)
library(tidymodels)
library(ggplot2)
library(tweedie)
library(forcats)

train_full <- fread("InsNova_train.csv")
train_full[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2, 
  veh_body = fct_lump_prop(veh_body, prop = 0.05),
  veh_age = factor(veh_age), 
  dr_age = factor(dr_age)
)]
set.seed(8051)
set_split <- initial_split(train_full, prob = 0.75)

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

# hunter_pred <- fread("data_test_withP2.csv")
# hunter_pred <- hunter_pred[order(id)]
# hunter_pred[, id := 1:nrow(hunter_pred)]
# fwrite(hunter_pred, "data_test_withP2_mod.csv")

# find optimal p for tweedie distribution
est_p <- tweedie.profile(claim_cost ~ veh_value + veh_value_sq + exposure + exposure_sq +
                           veh_body + factor(veh_age) + gender + area + factor(dr_age),
                         data = train_full, link.power = 0, do.smooth = TRUE, do.plot = TRUE)

library(tweedie)
library(statmod)

# fit many models with different var.power
tweedie_reg <- lapply(seq(1.1, 1.65, .01), function(p) {
  print(p)
  tweedie_model <- glm(claim_cost ~ veh_value + veh_value_sq + veh_body + factor(veh_age) + 
                         gender + area + factor(dr_age),
                       data = train_set, 
                       family = tweedie(var.power = p, link.power = 0),
                       offset = exposure)
  y_pred <- predict(tweedie_model, newdata = eval_set, type = "response")
  return(data.table(p = p, gini = MLmetrics::Gini(y_pred = y_pred, y_true = eval_set$claim_cost)))
})
tweedie_reg <- rbindlist(tweedie_reg)
tweedie_reg %>% 
  ggplot() +
  geom_point(aes(x = p, y = gini))

tweedie_reg[which.max(tweedie_reg$gini)]

tweedie_model <- glm(claim_cost ~ veh_value + veh_value_sq + veh_body + factor(veh_age) + 
                       gender + area + factor(dr_age),
                     data = train_full, 
                     family = tweedie(var.power = 1.64, link.power = 0),
                     offset = exposure)
summary(tweedie_model)

y_pred <- predict(tweedie_model, type = "response")
MLmetrics::Gini(y_pred = y_pred, y_true = train_full$claim_cost)

ggplot() + 
  geom_histogram(aes(x =  y_pred), binwidth = 50, fill = "red") +
  geom_histogram(data = train_full[claim_ind == 1], aes(x = claim_cost), binwidth = 50) +
  scale_x_continuous(limits = c(-25, 2500))


y_test <- predict(tweedie_model, newdata = test_set, type = "response")
tweedie_out <- data.frame(id = 1:nrow(test_set), claim_cost = y_test)
fwrite(tweedie_out, "tweedie_predictions.csv")