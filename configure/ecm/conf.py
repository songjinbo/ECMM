
params = {
    'model_name':"ECM",
    'field_count':22,
    'feature_count': 2**26,
    'dim': 8,
    'batch_size': 4096,
    'parallel' : 16,
    'learning_rate': 0.0004,
    'deep_layers':[128,64,32],
    'class_num': 3,
    'model_dir': 'model/ecm',
    'train_dir': './data/train',
    'test_dir': './data/test',
    'begin_day': '02',
    'end_day': '02',
}
