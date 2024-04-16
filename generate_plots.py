import os
import pandas as pd
from utils.visualization import plot_curves, bar_plot, grouped_bar_plot_reg
import matplotlib.pyplot as plt


results_dir = "evaluation/results/"
final_table_columns = ['loss', 'AUC', 'Precision', 'val_loss', 'val_AUC', 'val_Precision']

session_type = "reg"

SESSION_RESULTS_AUC = {}
SESSION_RESULTS_PREC = {}

# plot learning curves + bar plot
for session in os.listdir(results_dir):
    results_prec = {}
    results_auc = {}
    for history in os.listdir(results_dir + session + "/histories/"):
        df = pd.read_csv(results_dir + session + "/histories/" + history)

        # remove numbers of columns
        rename_d = {name: ''.join(i for i in name if not i.isdigit()) for name in df.columns}
        for k, v in rename_d.items():
            if v[-1] == "_":
                rename_d[k] = v[:-1]
        df.rename(rename_d, inplace=True, axis='columns')

        # drop unnecessary columns
        df = df[df.columns.intersection(final_table_columns)]

        # split DFs
        df_loss = df[[col for col in df.columns if "loss" in col]]
        df_auc = df[[col for col in df.columns if "AUC" in col]]
        df_prec = df[[col for col in df.columns if "Precision" in col]]

        # rename columns
        df_loss.rename({"loss": "Training",
                             "val_loss": "Validierung"},
                            inplace=True, axis='columns')

        df_auc.rename({"AUC": "Training",
                             "val_AUC": "Validierung"},
                            inplace=True, axis='columns')

        df_prec.rename({"Precision": "Training",
                             "val_Precision": "Validierung"},
                            inplace=True, axis='columns')

        df_list = [df_loss, df_prec, df_auc]

        results_prec[history[:-4]] = df_prec["Validierung"][len(df_prec)-30]
        results_auc[history[:-4]] = df_auc["Validierung"][len(df_auc)-30]
        SESSION_RESULTS_AUC[session] = results_auc
        SESSION_RESULTS_PREC[session] = results_prec

        plot_curves(df_list, fname=session + "_" + history[:-4])

    if session_type == "comb":
        bar_plot(results_prec, session_type=session_type, type="prec")
        bar_plot(results_auc, session_type=session_type, type="auc")

if session_type == "reg" or session_type == "base":
    grouped_bar_plot_reg(SESSION_RESULTS_AUC, session_type=session_type, type="auc")
    grouped_bar_plot_reg(SESSION_RESULTS_PREC, session_type=session_type, type="prec")