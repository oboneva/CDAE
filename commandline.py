import sys
import getopt
from torch.optim.adam import Adam
from torch.optim.adagrad import Adagrad
from configs import data_config, model_config, trainer_config


def parse_args(argv):
    try:
        opts, args = getopt.getopt(
            argv, "c:t:e:p:d:l:o:", ["cratio=", "trainbs=", "epochs=", "patience=", "datadir=", "lr=", "optim="])
    except getopt.GetoptError:
        print(
            'main.py -c <corruption_ratio> -t <train_batch_size> -e <epochs> -p <patience> -d <datadir> -l <lr> -o <optim>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'main.py -c <corruption_ratio> -t <train_batch_size> -e <epochs> -p <patience> -d <datadir> -l <lr> -o <optim>')
            sys.exit()
        elif opt in ("-c", "--cratio"):
            model_config.corruption_ratio = float(arg)
        elif opt in ("-t", "--trainbs"):
            data_config.train_batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            trainer_config.epochs = int(arg)
        elif opt in ("-p", "--patience"):
            trainer_config.patience = int(arg)
        elif opt in ("-d", "--datadir"):
            data_config.data_dir = arg
        elif opt in ("-l", "--lr"):
            trainer_config.learning_rate = float(arg)
        elif opt in ("-o", "--optim"):
            if arg == "Adam":
                trainer_config.optimizer = Adam
            else:
                trainer_config.optimizer = Adagrad
