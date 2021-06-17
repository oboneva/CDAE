import sys
import getopt
from configs import data_config, model_config, trainer_config


def parse_args(argv):
    try:
        opts, args = getopt.getopt(
            argv, "c:t:e:p:", ["cratio=", "trainbs=", "epochs=", "patience="])
    except getopt.GetoptError:
        print('main.py -c <corruption_ratio> -t <train_batch_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -c <corruption_ratio> -t <train_batch_size>')
            sys.exit()
        elif opt in ("-c", "--cratio"):
            model_config.corruption_ratio = float(arg)
        elif opt in ("-t", "--trainbs"):
            data_config.train_batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            trainer_config.epochs = int(opt)
        elif opt in ("-p", "--patience"):
            trainer_config.patience = int(opt)
