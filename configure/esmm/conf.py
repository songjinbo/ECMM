
params = {
    'mode':2,
    'model_name':"ESMM",
    'field_count':22,
    'feature_count': 2**26,
    'dim': 8,
    'batch_size': 4096,
    'parallel' : 16,
    'learning_rate': 0.0004,
    'deep_layers':{
        "etr": [128,64,32],
        "ctr": [128,64,32]
    },
    'class_num': 3,
    'model_dir': 'model/esmm',
    'train_dir': './data/train',
    'test_dir': './data/test',
    'begin_day': '02',
    'end_day': '02',
}
