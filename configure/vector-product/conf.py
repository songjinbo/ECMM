
params = {
    'model_name':'VECTOR_PRODUCT',
    'field_count':22,
    'user_feat_size':6,
    'item_feat_size':6,
    'feature_count': 2**26,
    'dim': 8,
    'batch_size': 4096,
    'parallel' : 16,
    'learning_rate': 0.0004,
    'deep_layers':[128,64,32],
    'model_dir': 'model/vector-product',
    'train_dir': './data/train',
    'test_dir': './data/test',
    'forbid_p2_metric': True,
    'begin_day': '02',
    'end_day': '02',
}
