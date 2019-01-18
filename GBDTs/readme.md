# GBDTs Hyperparameter Tuning

## XGBoost
```python
from skopt.space import Real, Integer
xgboost_params_space = [Real(1e-7, 1, prior='log-uniform', name='learning_rate'), 
                Real(0.5, 1.0, name='subsample'),
                Integer(2, 10, name='max_depth'), 
                Real(1e-16, 1e5, prior='log-uniform', name='min_child_weight'),
                Real(0.5, 1.0, name='colsample_bylevel'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0.5, 1.0, name='colsample_bynode'), 
                Real(1.0, 16.0, name='scale_pos_weight'), 
                Real(0.0, 100, name='bagging_temperature'), 
                Real(0.0, 100, name='random_strength'), 
                Real(0.0, 100, name='alpha'),
                Real(0.0, 100, name='lambda'),
                Real(0.0, 100, name='gamma')]
```

## LightGBM
```python
from skopt.space import Real, Integer
lgbm_params_space = [Real(1e-7, 1, prior='log-uniform', name='learning_rate'),
                Real(1, 1e7, prior='log-uniform', name='num_leaves'),
                Real(0.5, 1.0, name='feature_fraction'),
                Real(0.5, 1.0, name='bagging_fraction'), 
                Real(1e-16, 1e5, prior='log-uniform', name='min_child_weight'),
                Real(1e-16, 1e2, prior='log-uniform', name='min_child_samples'),
                Integer(2, 10, name='max_depth'),
                Real(0.5, 1.0, name='subsample'), 
                Real(0.5, 1.0, name='colsample_bylevel'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0.5, 1.0, name='colsample_bynode'), 
                Real(1.0, 16.0, name='scale_pos_weight'), 
                Real(0.0, 100, name='lambda_l1'),
                Real(0.0, 100, name='lambda_l2')]
```

## Catboost
```python
from skopt.space import Real, Integer
catboost_params_space = [Real(1e-7, 1, prior='log-uniform', name='learning_rate'), 
                Integer(2, 10, name='max_depth'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bylevel'),  
                Integer(1, 10, name='gradient_iterations'), 
                Real(1.0, 16.0, name='scale_pos_weight'), 
                Real(0.0, 1.0, name='bagging_temperature'), 
                Integer(1, 20, name='random_strength'), 
                Integer(2, 25, name='one_hot_max_size'),
                Real(1.0, 100, name='reg_lambda')]
```
