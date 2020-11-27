library(data.table)
library(tidymodels)
library(xgboost)
library(Matrix)

train_full <- fread("InsNova_train.csv")
train_full[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2,
  claim_ind = factor(claim_ind)
)]
set.seed(8051)
set_split <- initial_split(train_full, prob = 0.75)

train_set <- training(set_split)
eval_set <- testing(set_split)

test_set <- fread("InsNova_test.csv")
test_set[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2
)]
x <- train_set[, .(claim_ind, veh_value, exposure, veh_body, veh_age, gender, 
                   area, dr_age, veh_value_sq, exposure_sq)]

xgb_spec <- boost_tree(
  trees = 1000, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), x),
  learn_rate(),
  size = 100
)
xgb_wf <- workflow() %>%
  add_formula(claim_ind ~ .) %>%
  add_model(xgb_spec)
set.seed(8051)
vb_folds <- vfold_cv(x, strata = claim_ind)

doParallel::registerDoParallel()

set.seed(8051)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = vb_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE),
  metrics = metric_set(roc_auc)
)

collect_metrics(xgb_res)

xgb_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, mtry:sample_size) %>%
  pivot_longer(mtry:sample_size,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

show_best(xgb_res, "sens")

best_sens <- select_best(xgb_res, "sens")
best_sens

final_xgb <- finalize_workflow(
  xgb_wf,
  best_sens
)

library(vip)

final_xgb %>%
  fit(data = x) %>%
  pull_workflow_fit() %>%
  vip(geom = "point")
final_res <- last_fit(final_xgb, set_split)

final_model <- fit(final_xgb, x)

collect_metrics(final_res)
final_res %>%
  collect_predictions() %>%
  roc_curve(claim_ind, .pred_0) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(size = 1.5, color = "midnightblue") +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  )

final_res %>% 
  collect_predictions() %>% 
  conf_mat(truth = claim_ind, estimate = .pred_class)

y_pred <- predict(final_model, new_data = test_set)
predict(final_res, new_data = test_set)

eval_set[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2, 
  claim_ind = factor(claim_ind)
)]
final_xgb %>%
  fit(data = test_set)


# feature engineering -----
x_mat <- sparse.model.matrix(claim_ind ~ veh_value + veh_value_sq + exposure + exposure_sq + 
                               veh_body + veh_age + gender + area + dr_age - 1, data = x)
y <- x$claim_ind