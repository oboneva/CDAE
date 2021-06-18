from configs import model_config, data_config


def model_metadata():
    return "_CR_{}_BATCH_{}".format(model_config.corruption_ratio, data_config.train_batch_size)
