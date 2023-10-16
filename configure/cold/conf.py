
params = {
    'model_name':"COLD",
    'field_count':22,
    'feature_count': 2**26,
    'dim': 8,
    'batch_size': 4096,
    'parallel' : 16,
    'learning_rate': 0.0004,
    'deep_layers':[128,64,32],
    'class_num': 1,
    'model_dir': 'model/cold',
    'train_dir': './data/train',
    'test_dir': './data/test',
    'select_num': 11,
    'forbid_p2_metric': True,
    'begin_day': '02',
    'end_day': '02',
}
