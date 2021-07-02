import pandas as pd
from configs import model_config, data_config
import numpy as np
import matplotlib.pyplot as plt


def model_metadata():
    return "_CR_{}_BATCH_{}".format(model_config.corruption_ratio, data_config.train_batch_size)


def plot_results():
    map1_adam = np.array([0.5550, 0.5705, 0.4825, 0.4858, 0.4996, 0.5068])
    map5_adam = np.array([0.2338, 0.2428, 0.2033, 0.2053, 0.2106, 0.2137])
    map10_adam = np.array([0.1422, 0.1474, 0.1254, 0.1260, 0.1298, 0.1315])

    map1_ada = np.array([0.0898, 0.0907, 0.0849, 0.0858, 0.0822, 0.0860])
    map5_ada = np.array([0.0449, 0.0452, 0.0428, 0.0432, 0.0422, 0.0426])
    map10_ada = np.array([0.0304, 0.0307, 0.0288, 0.0292, 0.0284, 0.0287])

    index = ['0', '0.2', '0.4', '0.6', '0.8', '1']

    df1 = pd.DataFrame({'Adagrad': map10_ada, 'Adam': map10_adam}, index=index)

    fig, axs = plt.subplots(figsize=(8, 4))
    ax = df1.plot.bar(rot=0, ax=axs)

    # # Show the plot
    plt.title("MAP@10")
    fig.savefig("MAP@10.png")


if __name__ == "__main__":
    plot_results()
