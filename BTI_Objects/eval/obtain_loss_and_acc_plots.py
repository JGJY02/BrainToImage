import os
import pickle
import random
import sys
import matplotlib.pyplot as plt
import numpy as np

main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

save_path = "results/LSTM_DCGAN_512/" #"results/CNN_ACGAN_B_128_ori/" # results/CNN_Basic_ACGAN_512/ #
hist_file = "LSTM_all_stacked_signals_dual_512_64_ori/history_LSTM_all_stacked_signals_dual_512_64_ori_final.pkl"  #CNN_all_stacked_signals_dual_128_ori/history_CNN_all_stacked_signals_dual_128_ori_final.pkl

with open(f"trained_models/classifiers/All/000thresh/{hist_file}", "rb") as f:
    data = pickle.load(f)

# Check the type to ensure it's a dictionary
print(type(data))

# If it's a dictionary, print the keys
if isinstance(data, dict):
    print("Keys:", data.keys())
else:
    print("The loaded object is not a dictionary.")


def plot_graph(data, data1, data2, data3, lossPlot = True):
    # Create a figure and 3 subplots (3 rows, 1 column)
    iterations = range(len(data[data1[0]]))

    if lossPlot == True:
        # First plot
        fig, axs = plt.subplots(1, 3, figsize=(9, 10))  # You can also use (1, 3) for horizontal layout

        save_name = "loss_plots.png"
        axs[0].plot(iterations, data[data1[0]], color='blue')
        axs[0].plot(iterations, data[data1[1]], color='red')
        text_to_print_1 = f"{data1[2]} \n Final Train: {data[data1[0]][-1]:.3f} ~ Val: {data[data1[1]][-1]:.3f}"

        print(data[data1[0]][0])

        axs[0].set_title(text_to_print_1)

        # Second plot
        axs[1].plot(iterations, data[data2[0]], color='blue')
        axs[1].plot(iterations, data[data2[1]], color='red')
        text_to_print_2 = f"{data2[2]} \n Final Train: {data[data2[0]][-1]:.3f} ~ Val: {data[data2[1]][-1]:.3f}"
        print(data[data2[0]][0])

        axs[1].set_title(text_to_print_2)

        # Third plot
        axs[2].plot(iterations, data[data3[0]], color='blue')
        axs[2].plot(iterations, data[data3[1]], color='red')
        text_to_print_3 = f"{data3[2]} \n Final Train: {data[data3[0]][-1]:.3f} ~ Val: {data[data3[1]][-1]:.3f}"
        axs[2].set_title(text_to_print_3)

        print(data[data3[0]][0])

        # axs[2].set_ylim(0, 10)  # limit y to avoid extreme value
    else:
        fig, axs = plt.subplots(1, 2, figsize=(9, 10))  # You can also use (1, 3) for horizontal layout

        save_name = "acc_plots.png"
        # First plot
        axs[0].plot(iterations, data[data1[0]], color='blue')
        axs[0].plot(iterations, data[data1[1]], color='red')
        text_to_print_1 = f"{data1[2]} \n Final Train: {data[data1[0]][-1]*100:.2f} ~ Val: {data[data1[1]][-1]*100:.2f}"

        axs[0].set_title(text_to_print_1)

        # Second plot
        axs[1].plot(iterations, data[data2[0]], color='blue')
        axs[1].plot(iterations, data[data2[1]], color='red')
        text_to_print_2 = f"{data2[2]} \n Final Train: {data[data2[0]][-1]*100:.2f} ~ Val: {data[data2[1]][-1]*100:.2f}"

        axs[1].set_title(text_to_print_2)


    # Add spacing
    plt.tight_layout()
    # plt.show()
    save_file = save_path + save_name
    plt.savefig(save_file)  # You can also use .jpg, .pdf, .svg, etc.

def plot_total_loss_graph(data, data1, data2, data3, lossPlot = True):
    # Create a figure and 3 subplots (3 rows, 1 column)
    iterations = range(len(data[data1[0]]))

    # First plot
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))  # You can also use (1, 3) for horizontal layout

    save_name = "total_loss_plots.png"
    axs.plot(iterations, data[data1[0]], color='blue')
    axs.plot(iterations, data[data1[1]], color='red')
    text_to_print_1 = f"{data1[2]} \n Final Train: {data[data1[0]][-1]:.3f} ~ Val: {data[data1[1]][-1]:.3f}"

    print(data[data1[0]][0])

    axs.set_title(text_to_print_1)

    # Add spacing
    plt.tight_layout()
    # plt.show()
    save_file = save_path + save_name
    plt.savefig(save_file)  # You can also use .jpg, .pdf, .svg, etc.

plot_graph(data, ['loss', 'val_loss', 'Total Loss'], ['EEG_Class_Labels_loss', 'val_EEG_Class_Labels_loss', 'Class Label Loss'] , ['EEG_Class_type_Labels_loss', 'val_EEG_Class_type_Labels_loss', 'Subclass label Loss'])
plot_total_loss_graph(data, ['loss', 'val_loss', 'Total Loss'], ['EEG_Class_Labels_loss', 'val_EEG_Class_Labels_loss', 'Class Label Loss'] , ['EEG_Class_type_Labels_loss', 'val_EEG_Class_type_Labels_loss', 'Subclass label Loss'])



plot_graph(data, ['EEG_Class_Labels_accuracy', 'val_EEG_Class_Labels_accuracy', 'Class Label Accuracy'] , ['EEG_Class_type_Labels_accuracy', 'val_EEG_Class_type_Labels_accuracy', 'Subclass label Accuracy'], 0,  False)