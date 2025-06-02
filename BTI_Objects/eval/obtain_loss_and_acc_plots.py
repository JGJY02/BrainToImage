import os
import pickle
import random
import sys
import matplotlib.pyplot as plt
import numpy as np

main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

with open("trained_models/classifiers/All/000thresh/CNN_all_stacked_signals_dual_128_ori/history_CNN_all_stacked_signals_dual_128_ori_final.pkl", "rb") as f:
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
    if lossPlot == True:
        fig, axs = plt.subplots(1, 3, figsize=(8, 10))  # You can also use (1, 3) for horizontal layout
        iterations = range(len(data[data1[0]]))
        # First plot
        axs[0].plot(iterations, data[data1[0]], color='blue')
        axs[0].plot(iterations, data[data1[1]], color='red')
        axs[0].set_title(data1[2])

        # Second plot
        axs[1].plot(iterations, data[data2[0]], color='blue')
        axs[1].plot(iterations, data[data2[1]], color='red')
        axs[1].set_title(data2[2])

        # Third plot
        axs[2].plot(iterations, data[data3[0]], color='blue')
        axs[2].plot(iterations, data[data3[1]], color='red')
        axs[2].set_title(data3[2])
        # axs[2].set_ylim(0, 10)  # limit y to avoid extreme value
    else:
        fig, axs = plt.subplots(1, 2, figsize=(8, 10))  # You can also use (1, 3) for horizontal layout
        iterations = range(len(data[data1[0]]))
        # First plot
        axs[0].plot(iterations, data[data1[0]], color='blue')
        axs[0].plot(iterations, data[data1[1]], color='red')
        axs[0].set_title(data1[2])

        # Second plot
        axs[1].plot(iterations, data[data2[0]], color='blue')
        axs[1].plot(iterations, data[data2[1]], color='red')
        axs[1].set_title(data2[2])


    # Add spacing
    plt.tight_layout()
    plt.show()

plot_graph(data, ['loss', 'val_loss', 'Total Loss'], ['EEG_Class_Labels_loss', 'val_EEG_Class_Labels_loss', 'Class Label Loss'] , ['EEG_Class_type_Labels_loss', 'val_EEG_Class_type_Labels_loss', 'Subclass label Loss'])

plot_graph(data, ['EEG_Class_Labels_accuracy', 'val_EEG_Class_Labels_accuracy', 'Class Label Accuracy'] , ['EEG_Class_type_Labels_accuracy', 'val_EEG_Class_type_Labels_accuracy', 'Subclass label Accuracy'], 0,  False)