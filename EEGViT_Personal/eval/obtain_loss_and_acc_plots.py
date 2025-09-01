import os
import pickle
import random
import sys
import matplotlib.pyplot as plt
import numpy as np

main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

save_path = "trained_models/Transformer_512_dual/" #"results/CNN_ACGAN_B_128_ori/" # 
hist_file = "trained_models/Transformer_512_dual/results_HyperparamTuning.npy"  #CNN_all_stacked_signals_dual_128_ori/history_CNN_all_stacked_signals_dual_128_ori_final.pkl
desc = "hyperparamTuning"
data = np.load(hist_file, allow_pickle = True).item()

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

        save_name = f"loss_plots_{desc}.png"
        print(data[data1[0]])
        axs[0].plot(iterations, data[data1[0]], color='blue')
        axs[0].plot(iterations, data[data1[1]], color='red')
        text_to_print_1 = f"{data1[2]} \n Final Train: {data[data1[0]][-1]:.3f} ~ Val: {data[data1[1]][-1]:.3f}"
        axs[0].set_title(text_to_print_1)

        # Second plot
        print(data[data2[0]])

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

        save_name = f"acc_plots_{desc}.png"
        # First plot
        print(data[data1[0]])
        print(data1[0])
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

    save_name = f"total_loss_plots_{desc}.png"
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


plot_graph(data, ['total_train_loss', 'total_val_loss', 'Total Loss'], ['train_loss_class', 'val_loss_class', 'Class Label Loss'] , ['train_loss_type', 'val_loss_type', 'Subclass label Loss'])
plot_total_loss_graph(data, ['total_train_loss', 'total_val_loss', 'Total Loss'], ['train_loss_class', 'val_loss_class', 'Class Label Loss'] , ['train_loss_type', 'val_loss_type', 'Subclass label Loss'])


plot_graph(data, ['train_acc_class', 'val_acc_class', 'Class Label Accuracy'] , ['train_acc_type', 'val_acc_type', 'Subclass label Accuracy'], 0,  False)