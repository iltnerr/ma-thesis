import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cycler
import os

def plot_kpt_map(image_t, map_t, predictions=None, greyscale=False, filename=None):
    """
    Use x, y pairs from the dataset to plot an input image with the corresponding keypoints.
    """
    if not image_t.shape[0] == 1 and not map_t.shape[0] == 1:
        # use first item of the batch if batch_size > 1
        print("Batch Size > 1. Using only the first item of the batch to plot the keypoint map.")
        image_t = tf.expand_dims(tf.unstack(image_t)[0], axis=0)
        map_t = tf.expand_dims(tf.unstack(map_t)[0], axis=0)
        if predictions is not None:
            predictions = tf.expand_dims(tf.unstack(predictions)[0], axis=0)

    # unbatch + convert to numpy + greyscale
    if len(image_t.shape) == 4:
        image_t = tf.squeeze(image_t, axis=0)

    if len(map_t.shape) > 2:
        map_t = tf.squeeze(map_t)

    map = map_t.numpy()

    if greyscale and image_t.shape[2] == 3:
        image_t = tf.image.rgb_to_grayscale(image_t)

    image = image_t.numpy()*255
    image = image.astype(np.uint8)

    if not greyscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # draw keypoints
    thickness = 1
    # ground truth
    radius = 5
    color = (255, 0, 0) # Blue
    indexes = np.argwhere(map == 1)
    kpts = pd.DataFrame(indexes, columns=['y', 'x'])
    for index, row in kpts.iterrows():
        center_coordinates = (int(row["x"]), int(row["y"]))
        image = cv2.circle(image, center_coordinates, radius, color, thickness)

    # predictions
    if predictions is not None:
        predictions = tf.squeeze(predictions).numpy()

        radius = 1
        color = (0, 0, 255)  # Red
        indexes_pred = np.argwhere(predictions == 1)
        kpts_pred = pd.DataFrame(indexes_pred, columns=['y', 'x'])
        for index, row in kpts_pred.iterrows():
            center_coordinates = (int(row["x"]), int(row["y"]))
            image = cv2.circle(image, center_coordinates, radius, color, thickness)

    if filename is not None:
        cv2.imwrite("plots/predictions/" + filename + ".png", image)
    else:
        cv2.imshow("Keypoint_Map", image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

def plot_histories():
    """
    plot histories after training to better understand the training process.
    """
    def plot_history(df, fname):

        # remove numbers of columns
        rename_d = {name: ''.join(i for i in name if not i.isdigit()) for name in df.columns}
        for k, v in rename_d.items():
            if v[-1] == "_":
                rename_d[k] = v[:-1]
        df.rename(rename_d, inplace=True, axis='columns')

        # split DFs
        df_loss = df[[col for col in df.columns if "loss" in col]]
        df_iou = df[[col for col in df.columns if "IOU" in col]]
        df_auc = df[[col for col in df.columns if "AUC" in col]]
        df_prec_rec = df[[col for col in df.columns if "Precision" in col or "Recall" in col]]

        SMALL_SIZE = 10
        BIGGER_SIZE = 10
        TICK_SIZE = 0.2

        plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE - 2)  # legend fontsize
        plt.rcParams['axes.linewidth'] = 0.1

        fig, axes = plt.subplots(nrows=2, ncols=2)

        df_loss.plot(ax=axes[0, 0])
        df_prec_rec.plot(ax=axes[0, 1])
        df_iou.plot(ax=axes[1, 0])
        df_auc.plot(ax=axes[1, 1])

        # plt.tight_layout()
        plt.savefig("plots/histories/" + fname + ".pdf")
        plt.clf()

    directory = os.fsencode("models/histories/")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if "history" in filename:
            df = pd.read_csv("models/histories/" + filename)

            filename = filename.split("-history")[0]

            plot_history(df, filename)

def plot_curves(df_list, fname):
    """used to plot learning curves. this is an updated version of plot_history()"""

    SMALL_SIZE = 10
    BIGGER_SIZE = 10
    TICK_SIZE = 0.2

    MAX_EPOCHS = len(df_list[0])

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE - 2)  # legend fontsize
    plt.rc('lines', linewidth=2)
    plt.rcParams['axes.linewidth'] = 0.1

    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                     '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')

    fig, axes = plt.subplots(3)
    fig.suptitle(fname)

    for idx in range(0, 3):
        df_list[idx].plot(ax=axes[idx])
        axes[idx].axvline(x=MAX_EPOCHS-30, color='darkorange', linestyle='dashed', label="Early-Stopping", lw=1, alpha=0.5)
        axes[idx].legend()

    plt.xlabel("Epochen")

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Precision")
    axes[2].set_ylabel("AUC")

    plt.tight_layout()
    plt.savefig("plots/learning_curves/" + fname + ".pdf")
    plt.clf()

def bar_plot(results, session_type, type):
    """
    Bar Plot of Precision
    """

    SMALL_SIZE = 10
    BIGGER_SIZE = 10
    TICK_SIZE = 0.2

    if type == "prec":
        colors = cycler('color',
                        ['#3062c7', '#3388BB', '#9988DD',
                         '#EECC55', '#88BB44', '#FFBBBB'])

    if type == "auc":
        colors = cycler('color',
                        ['#c79530', '#3388BB', '#9988DD',
                         '#EECC55', '#88BB44', '#FFBBBB'])

    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=SMALL_SIZE, facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE - 2)  # legend fontsize
    plt.rc('lines', linewidth=2)
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')

    fig, axes = plt.subplots(figsize=(6, 4))

    metric = "Precision " if type == "prec" else "AUC "
    experiment = "C" if session_type == "comb" else "EXP-PLATZHALTER"

    fig.suptitle(metric + "für Experiment " + experiment)

    axes.bar(["C-1", "C-2", "C-3", "C-4", "C-5", "C-6"], results.values(), edgecolor='white', width=0.3, align='center')
    axes.grid(axis="x", color='gray', zorder=0)
    axes.set_axisbelow(True)

    axes = plt.gca()
    if type == "prec":
        axes.set_ylabel("Precision")
    if type == "auc":
        axes.set_ylabel("AUC")
        axes.set_ylim([0.45, 0.8])
    axes.set_xlabel('Modell')

    # plt.tight_layout()
    plt.savefig("plots/barplots/" + session_type + "_" + type + ".pdf")
    plt.clf()

def grouped_bar_plot_reg(results, session_type, type):

    # rearrange data per model
    data_per_model = {}
    for model in results["canny"].keys():
        data_per_model[model] = []

        for session in results.keys():
            data_per_model[model].append(results[session][model])

    SMALL_SIZE = 10
    BIGGER_SIZE = 10
    TICK_SIZE = 0.2

    colors = cycler('color',
                    ['#3062c7', '#3388BB', '#9988DD',
                     '#EECC55', '#88BB44', '#FFBBBB'])

    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=SMALL_SIZE, facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE - 2)  # legend fontsize
    plt.rc('lines', linewidth=2)
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')

    barWidth = 0.2

    # set heights of bars
    bars1 = [data[0] for data in data_per_model.values()]
    bars2 = [data[1] for data in data_per_model.values()]
    bars3 = [data[2] for data in data_per_model.values()]
    bars4 = [data[3] for data in data_per_model.values()]
    bars_l = [bars1, bars2, bars3, bars4]

    # replace zero-valued bars with very low values to make the plot look nicer
    for bar in bars_l:
        for n, i in enumerate(bar):
            if i == 0:
                bar[n] = 0.01

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    fig, axes = plt.subplots(figsize=(6, 4))

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Canny Kantenbilder (C)')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Natürliche Bilder (RGB)')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='G. T. Segmentierungen (S)')
    plt.bar(r4, bars4, color='#3062c7', width=barWidth, edgecolor='white', label='Lernbasierte Kantenbilder (K)')

    axes.grid(axis="x", color='gray', zorder=0)

    # Add xticks on the middle of the group bars
    if session_type == "reg":
        plt.xticks([r + barWidth for r in range(len(bars1))], ['B-1', 'B-2', 'B-3', 'B-4', 'B-5', 'B-6', 'B-7', 'B-8'])
    else:
        plt.xticks([r + barWidth for r in range(len(bars1))], ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6'])

    axes = plt.gca()
    if type == "prec":
        axes.set_ylabel("Precision")
    if type == "auc":
        axes.set_ylabel("AUC")
        axes.set_ylim([0.45, 0.8])
    axes.set_xlabel('Modell')

    metric = "Precision für Experiment " if type == "prec" else "AUC für Experiment "
    experiment = "A" if session_type == "base" else "B"
    fig.suptitle(metric + experiment)

    # Create legend & Show graphic
    plt.legend()
    # plt.tight_layout()
    plt.savefig("plots/barplots/" + session_type + "_" + type + ".pdf")
    plt.clf()