CatBoost_params = {
    'iterations': 50000,
    'use_best_model': True,
    'eval_metric': 'RMSE',
    'od_type': 'Iter',
    'od_wait': 200,
    'verbose': 200,
    'learning_rate': .01,
    'depth': 7,
    'bagging_temperature': .7,
}