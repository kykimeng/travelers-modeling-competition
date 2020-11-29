library(data.table)
library(tidymodels)
library(xgboost)
library(Matrix)
library(forcats)
library(MLmetrics)

train_full <- fread("InsNova_train.csv")
# train_full[, `:=` (
#   veh_value_sq = veh_value^2,
#   exposure_sq = exposure^2,
#   claim_ind = factor(claim_ind)
# )]
set.seed(8051)
set_split <- initial_split(train_full, prob = 0.75)

train_set <- training(set_split)
eval_set <- testing(set_split)

test_set <- fread("InsNova_test.csv")
test_set[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2
)]

step_recipe <- recipe(claim_ind ~ veh_value + exposure + veh_body + veh_age + gender + area + dr_age,
                      data = train_full) %>% 
  # step_num2factor(claim_ind, transform = function(x) x + 1, levels = c('no', 'yes')) %>% 
  step_mutate(veh_body = fct_lump_prop(veh_body, prop = 0.05), 
              dr_age = factor(dr_age), 
              veh_age = factor(veh_age), 
              wt = ifelse(claim_ind == "yes", 20, 1)) %>% 
  step_log(veh_value, offset = 1) %>% 
  step_dummy(all_nominal(), -claim_ind) %>% 
  step_interact(~ starts_with("veh_body"):starts_with("gender") + 
                  starts_with("veh_body"):starts_with("area") +
                  starts_with("veh_body"):veh_value + 
                  starts_with("veh_body"):exposure +
                  starts_with("veh_body"):starts_with("veh_age") + 
                  starts_with("veh_body"):starts_with("dr_age") +
                  starts_with("gender"):starts_with("area") + 
                  starts_with("gender"):veh_value +
                  starts_with("gender"):exposure + 
                  starts_with("gender"):starts_with("veh_age") +
                  starts_with("gender"):starts_with("dr_age") + 
                  starts_with("area"):veh_value +
                  starts_with("area"):exposure + 
                  starts_with("area"):starts_with("veh_age") + 
                  starts_with("area"):starts_with("dr_age") +
                  veh_value:exposure + 
                  veh_value:starts_with("veh_age") + 
                  veh_value:starts_with("dr_age") + 
                  exposure:starts_with("veh_age") +
                  exposure:starts_with("dr_age") + 
                  starts_with("veh_age"):starts_with("dr_age")) %>% 
  step_poly(veh_value, exposure, degree = 3) %>% 
  step_zv(all_predictors(), -all_outcomes(), -wt, -starts_with("exposure"), -starts_with("veh_value")) %>%
  step_nzv(all_predictors(), -all_outcomes(), -wt, -starts_with("exposure"), -starts_with("veh_value"),
           freq_cut = 99/1) %>%
  step_corr(all_numeric(), -all_outcomes(), -wt, -starts_with("exposure"), -starts_with("veh_value")) %>%
  prep()

train_prep <- juice(step_recipe)
set.seed(8051)
set_split <- initial_split(train_prep, prob = 0.75, strata = "claim_ind")

train_set <- training(set_split)
eval_set <- testing(set_split)
test_set_prep <- bake(step_recipe, new_data = test_set)

dtrain <- xgb.DMatrix(data = train_set %>% select(-claim_ind, -wt) %>% as.matrix(),
                       label = train_set$claim_ind)
dtest <- xgb.DMatrix(data = eval_set %>% select(-claim_ind, -wt) %>% as.matrix(), 
                     label = eval_set$claim_ind)

params <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eta = 0.3,
               gamma = 0,
               max_depth = 6,
               min_child_weight = 1,
               subsample = 1,
               colsample_bytree = 1)
xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                metrics = "auc",
                weight = train_set$wt,
                nrounds = 100, 
                nfold = 10, 
                showsd = T, 
                stratified = T, 
                print_every_n = 10, 
                early_stopping_rounds = 20,
                maximize = F)

bstDMatrix <- xgboost(data = x_train, 
                      weight = wt,
                      eval_metric = "auc",
                      nrounds = 10,
                      objective = "binary:logistic")

y_pred <- predict(xgbcv, eval_set %>% select(-claim_ind) %>% as.matrix())

# regression ----
step_recipe <- recipe(claim_cost ~ exposure + veh_value + veh_body + veh_age + gender + area + dr_age,
                      data = train_full) %>% 
  step_mutate(veh_body = fct_lump_prop(veh_body, prop = 0.05), 
              dr_age = factor(dr_age), 
              veh_age = factor(veh_age)) %>% 
  step_log(veh_value, offset = 1) %>% 
  step_mutate(veh_value_sq = veh_value^2, 
              exposure_sq = exposure^2) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_poly(veh_value, degree = 3) %>% 
  step_zv(all_predictors(), -all_outcomes()) %>%
  prep()

train_prep <- juice(step_recipe)
set.seed(8051)
set_split <- initial_split(train_prep, prob = 0.75)

train_set <- training(set_split)
eval_set <- testing(set_split)

dtrain <- xgb.DMatrix(data = train_set %>% select(-claim_cost) %>% as.matrix(),
                      label = train_set$claim_cost)
setinfo(dtrain, "base_margin", train_set$exposure)
dtest <- xgb.DMatrix(data = eval_set %>% select(-claim_cost) %>% as.matrix(), 
                     label = eval_set$claim_cost)
setinfo(dtest, "base_margin", eval_set$exposure)

set.seed(8051)
xgb_grid <- grid_latin_hypercube(
  trees(c(10, 1000)),
  tree_depth(),
  loss_reduction(),
  sample_size = sample_prop(),
  learn_rate(),
  size = 30
)

out_gini <- sapply(1:30, function(i) {
  p <- xgb_grid[i,]
  params <- list(
    objective = 'reg:tweedie',
    eval_metric = 'rmse',
    tweedie_variance_power = 1.565919,
    max_depth = p$tree_depth,
    eta = p$learn_rate,
    gamma = p$loss_reduction)
  
  xgbcv <- xgb.cv(params = params, 
                  data = dtrain, 
                  metrics = "rmse",
                  nrounds = p$trees, 
                  nfold = 10, 
                  showsd = T, 
                  stratified = T, 
                  print_every_n = 10, 
                  early_stopping_rounds = 20,
                  maximize = F)
  
  bst <- xgb.train(
    data = dtrain,
    params = params,
    maximize = FALSE,
    watchlist = list(val = dtest, train = dtrain),
    nrounds = xgbcv$best_ntreelimit)
  preds <- predict(bst, dtest)
  
  return(Gini(preds, eval_set$claim_cost))
})

p <- xgb_grid[which.max(out_gini),]
params <- list(
  objective = 'reg:tweedie',
  eval_metric = 'rmse',
  tweedie_variance_power = 1.565919,
  max_depth = p$tree_depth,
  eta = p$learn_rate,
  gamma = p$loss_reduction)

xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                metrics = "rmse",
                nrounds = p$trees, 
                nfold = 10, 
                showsd = T, 
                stratified = T, 
                print_every_n = 10, 
                early_stopping_rounds = 20,
                maximize = F)
dtrain_full <- xgb.DMatrix(data = train_prep %>% select(-claim_cost) %>% as.matrix(),
                           label = train_prep$claim_cost)
setinfo(dtrain_full, "base_margin", train_prep$exposure)
bst <- xgb.train(
  data = dtrain,
  params = params,
  maximize = FALSE,
  nrounds = xgbcv$best_ntreelimit)
preds <- predict(bst, dtest)
Gini(preds, eval_set$claim_cost)

bst <- xgb.train(
  data = dtrain_full,
  params = params,
  maximize = FALSE,
  nrounds = xgbcv$best_ntreelimit)

dtest_final <- xgb.DMatrix(data = test_set_prep %>% as.matrix())
preds <- predict(bst, dtest_final)
fwrite(data.table(id = 1:nrow(test_set), claim_cost = preds), "predictions_xgb_tweedie.csv")

# linear regression ----
out_gini <- sapply(1:30, function(i) {
  p <- xgb_grid[i,]
  params <- list(
    objective = 'reg:squaredlogerror',
    eval_metric = 'rmse',
    tweedie_variance_power = 1.565919,
    max_depth = p$tree_depth,
    eta = p$learn_rate,
    gamma = p$loss_reduction)
  
  xgbcv <- xgb.cv(params = params, 
                  data = dtrain, 
                  metrics = "rmse",
                  nrounds = p$trees, 
                  nfold = 10, 
                  showsd = T, 
                  stratified = T, 
                  print_every_n = 10, 
                  early_stopping_rounds = 20,
                  maximize = F)
  
  bst <- xgb.train(
    data = dtrain,
    params = params,
    maximize = FALSE,
    watchlist = list(val = dtest, train = dtrain),
    nrounds = xgbcv$best_ntreelimit)
  preds <- predict(bst, dtest)
  
  return(Gini(preds, eval_set$claim_cost))
})

p <- xgb_grid[which.max(out_gini),]
params <- list(
  objective = 'reg:squaredlogerror',
  eval_metric = 'rmse',
  tweedie_variance_power = 1.565919,
  max_depth = p$tree_depth,
  eta = p$learn_rate,
  gamma = p$loss_reduction)

xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                metrics = "rmse",
                nrounds = p$trees, 
                nfold = 10, 
                showsd = T, 
                stratified = T, 
                print_every_n = 10, 
                early_stopping_rounds = 20,
                maximize = F)
dtrain_full <- xgb.DMatrix(data = train_prep %>% select(-claim_cost) %>% as.matrix(),
                           label = train_prep$claim_cost)
setinfo(dtrain_full, "base_margin", train_prep$exposure)
bst <- xgb.train(
  data = dtrain,
  params = params,
  maximize = FALSE,
  nrounds = xgbcv$best_ntreelimit)
preds <- predict(bst, dtest)
Gini(preds, eval_set$claim_cost)

bst <- xgb.train(
  data = dtrain_full,
  params = params,
  maximize = FALSE,
  nrounds = xgbcv$best_ntreelimit)

dtest_final <- xgb.DMatrix(data = test_set_prep %>% as.matrix())
preds <- predict(bst, dtest_final)
fwrite(data.table(id = 1:nrow(test_set), claim_cost = preds), "predictions_xgb_tweedie.csv")



# tidymodels ----
AUC(y_pred, eval_set$claim_ind)

data.frame(claim_ind = factor(eval_set$claim_ind), y_pred = 1-y_pred) %>% 
  roc_curve(claim_ind, y_pred) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(size = 1.5, color = "midnightblue") +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  )

xgb_spec <- boost_tree(
  trees = tune(), 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>% 
  set_engine("xgboost", weight = "wt") %>% 
  set_mode("classification")
xgb_grid <- grid_latin_hypercube(
  trees(c(10, 1000)),
  tree_depth(),
  min_n(c(5, 25)),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), train_set),
  learn_rate(),
  size = 10
)
xgb_wf <- workflow() %>%
  add_formula(claim_ind ~ .) %>%
  add_model(xgb_spec)
set.seed(8051)
vb_folds <- vfold_cv(train_set, strata = claim_ind)

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

show_best(xgb_res, "roc_auc")

best_auc <- select_best(xgb_res, "roc_auc")
best_auc

final_xgb <- finalize_workflow(
  xgb_wf,
  best_auc
)

library(vip)

final_xgb %>%
  fit(data = train_set) %>%
  pull_workflow_fit() %>%
  vip(geom = "point")
final_res <- last_fit(final_xgb, set_split)

final_model <- fit(final_xgb, train_set)

collect_metrics(final_res)
final_res %>%
  collect_predictions() %>%
  roc_curve(claim_ind, .pred_no) %>%
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

y_pred <- predict(final_model, new_data = test_prep, type = "prob")
y_pred_class <- (y_pred$.pred_yes >= .068)*1

data.frame(y_pred = y_pred$.pred_no, y_true = test_prep$claim_ind) %>% 
  roc_curve(y_true, y_pred) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(size = 1.5, color = "midnightblue") +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  )

predict(final_res, new_data = test_set)

eval_set[, `:=` (
  veh_value_sq = veh_value^2,
  exposure_sq = exposure^2, 
  claim_ind = factor(claim_ind)
)]
final_xgb %>%
  fit(data = test_set)

