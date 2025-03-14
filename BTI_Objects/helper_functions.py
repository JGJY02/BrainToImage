import csv
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycatch22 as catch22
from scipy.signal import butter, filtfilt, iirnotch, welch, firwin, lfilter



## HCTSA Catch22 feature set.
def compute_features(x):
    res = catch22.catch22_all(x,catch24=True,short_names=True)
    return res['values'] # just return the values

# Loads series data from a CSV file into a pandas DataFrame.
def load_series_data(file_path,isHeader=None):
    """ Loads series data from a CSV file into a pandas DataFrame.


    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the series data.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=isHeader)

    # Reshape the DataFrame to have one column per series row
    series_data = df.set_index(0).T

    # Reset the index to have a default integer index
    series_data = series_data.reset_index(drop=True)

    return series_data

# Parse a string of comma-separated values into a dictionary
def parse_text(text, params=None):
    """    Parse a string of comma-separated values into a dictionary with specified parameters.

    Args:
        text (str): A string of comma-separated values representing the parameters and their values.

    Returns:
        dict: A dictionary with parameter names as keys and their corresponding values as lists.

    The expected parameters and their number of comma-separated values are:
        [digit_label]: 1
        [digit_label_png]: 784
        [EEGdata_TP9]: 512,
        [EEGdata_AF7]: 512,
        [EEGdata_AF8]: 512,
        [EEGdata_TP10]: 512,
        [PPGdata_PPG1]: 512
        [PPGdata_PPG2]: 512
        [PPGdata_PPG3]: 512
        [Accdata_X]: 512
        [Accdata_Y]: 512
        [Accdata_Z]: 512
        [Gyrodata_X]: 512
        [Gyrodata_Y]: 512
        [Gyrodata_Z]: 512

    If the total number of comma-separated values does not match the expected count (6420),
    a warning message will be printed.
    """
    if not params:
        # Define the parameters and their expected number of values
        parameters = dataset_formats.Muse2_v017_params
    elif isinstance(params,dict):
        parameters = params
    else:
        print("parameters should be a dictionary of data headings and word counts.")
        return None

    # Split the text into comma-separated values
    values = text.split(",")

    # Check if the total number of values matches the expected count

    expected_count = sum(item["size"] for item in parameters.values()) #sum(parameters.values()["size"])
    if len(values) != expected_count:
        print(f"Warning: Expected {expected_count} values, but got {len(values)}")

    # Initialize the result dictionary
    result = {}

    # Iterate over the parameters and assign values
    start_index = 0
    for param, meta in parameters.items():
        end_index = start_index + meta["size"]
        if meta["size"] > 1:
            result[param] = np.array([meta["type"](item) for item in values[start_index:end_index]],dtype=meta["type"])
        else:
            result[param] = meta["type"](values[start_index:end_index][0])
        start_index = end_index

    return result

# Read a CSV file and process each row using the parse_text method.
def process_csv_file(file_path, row_range=None):
    """     Read a CSV file and process each row using the parse_text method.

    Args:
        file_path (str): The path to the CSV file.
        row_range (int or tuple, optional): The number of rows to process or a range of rows.
            If an integer is provided, it specifies the number of rows to process.
            If a tuple is provided, it specifies the start and end rows (inclusive) to process.
            If None, all rows will be processed.

    Returns:
        list: A list of dictionaries, where each dictionary represents a processed row.
    """
    processed_rows = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        row_count = 0

        for row in csv_reader:
            # Check if the row is within the specified range
            if row_range is None and row_count == 0:
                continue
            if row_range is None or (
                (isinstance(row_range, int) and row_count < row_range) or
                (isinstance(row_range, tuple) and row_range[0] <= row_count < row_range[1])
            ):
                # Join the row elements into a comma-separated string
                text = ','.join(row)

                # Process the text using the parse_text method
                parsed_row = parse_text(text)

                # Append the parsed row to the list
                processed_rows.append(parsed_row)

            row_count += 1

            # Break the loop if the specified number of rows has been processed
            if isinstance(row_range, int) and row_count >= row_range:
                break
            if isinstance(row_range, tuple) and row_count >= row_range[1]:
                break

    return processed_rows

# Save the 'EEGdata_TP9' as example parameter as a .dat file.
def save_eeg_data_v2(path, parsed_row, id, save_data=None, dataset="Train"):
    """     Save the 'EEGdata_TP9' as example parameter as a .dat file.

    Args:
        parsed_row (dict): The dictionary returned by the parse_text method.
        save_data (None | string) Save the <string> parameter as a .dat file.

    Returns:
        str: The path to the created .dat file.
    """
    # Extract the required parameters from the parsed row
    dataset = dataset
    origin = parsed_row.get('digit_label', [''])
    origin_png = parsed_row.get('digit_label_png',[''])
    ts = int(datetime.now().timestamp())
    if save_data:
        _data = parsed_row.get(save_data, [])
        data_name = save_data
    else:
        _data = parsed_row.get('EEGdata_TP9', [])
        data_name = 'EEGdata_TP9'

    # Construct the filename
    filename = f"{data_name}_{dataset}_{origin}_{ts}_{id}.dat"
    file_path = os.path.join(path, filename)

    # Create the 'data' directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Save the EEG data to the .dat file
    with open(file_path, 'w') as file:
        for value in _data:
            file.write(f"{value}\n")

    return file_path

# Save the 'EEGdata_TP9' as example parameter as a .dat file.
def save_eeg_data(parsed_row, save_data=None):
    """     Save the 'EEGdata_TP9' as example parameter as a .dat file.

    Args:
        parsed_row (dict): The dictionary returned by the parse_text method.
        save_data (None | string) Save the <string> parameter as a .dat file.

    Returns:
        str: The path to the created .dat file.
    """
    # Extract the required parameters from the parsed row
    dataset = parsed_row.get('dataset', [''])[0]
    origin = parsed_row.get('origin', [''])[0]
    digit_event = parsed_row.get('digit_event', [''])[0]
    timestamp = parsed_row.get('timestamp', [''])[0]
    if save_data:
        _data = parsed_row.get(save_data, [])
        data_name = save_data
    else:
        _data = parsed_row.get('EEGdata_TP9', [])
        data_name = 'EEGdata_TP9'

    # Construct the filename
    filename = f"{data_name}_{dataset}_{origin}_{digit_event}_{timestamp}.dat"
    file_path = os.path.join('data', filename)

    # Create the 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save the EEG data to the .dat file
    with open(file_path, 'w') as file:
        for value in _data:
            file.write(f"{value}\n")

    return file_path

# Read filenames in a target directory with a target extension
def process_filenames(target_dir, target_ext='.dat', delimiter='_', output_file='parsed_filenames.txt', positions=None, filename_filter=None):
    """
    Read filenames in a target directory with a target extension, parse them based on a delimiter,
    and write a line to a text file for each parsed filename with the complete filename and
    comma-separated values for parmeters embedded in the filename from the parsed filename at specified positions.

    Args:
        target_dir (str): The path to the target directory.
        target_ext (str, optional): The target file extension. Default is '.dat'.
        delimiter (str, optional): The delimiter character used to parse the filenames. Default is '_'.
        output_file (str, optional): The name of the output text file. Default is 'parsed_filenames.txt'.
        positions (list, optional): A list of zero-based positions to extract values from the parsed filename.
            If None, all values will be included.

    Returns:
        str: The path to the output text file.
    """
    output_path = os.path.join(target_dir, output_file)

    with open(output_path, 'w') as output_file:
        for filename in os.listdir(target_dir):
            if filename.endswith(target_ext):
                if filename_filter is None or re.search(filename_filter, filename):
                    parsed_filename = filename.split(delimiter)
                    complete_filename = filename
                    values = parsed_filename if positions is None else [parsed_filename[pos] for pos in positions]
                    output_line = f"{complete_filename}  {','.join(values)}\n"
                    output_file.write(output_line)

    return output_path


def binary_array_to_image(binary_array):
    """
    Convert a binary NumPy array of shape (784,) to a 28x28 black and white image.

    Parameters:
    binary_array (numpy.ndarray): A 1D array of shape (784,) containing binary values (0 or 1).

    Returns:
    None: Displays the image using matplotlib.
    """
    # Ensure the input is a NumPy array
    binary_array = np.array(binary_array)

    # Check if the input array has the correct shape
    if binary_array.shape != (784,):
        raise ValueError("Input array must have shape (784,)")

    # Check if the input array contains only binary values
    if not np.all(np.isin(binary_array, [0, 1])):
        raise ValueError("Input array must contain only binary values (0 or 1)")

    # Reshape the array to 28x28
    image = binary_array.reshape(28, 28)

    # Create a new figure
    plt.figure(figsize=(5, 5))

    # Display the image
    plt.imshow(image, cmap='binary')
    plt.axis('off')  # Turn off axis labels
    plt.title('28x28 Black and White Image')

    # Show the plot
    plt.show()

# Example usage:
# Assuming you have a binary NumPy array called 'my_array' of shape (784,)
# binary_array_to_image(my_array)

def grayscale_array_to_image(grayscale_array):
    """
    Convert a grayscale NumPy array of shape (784,) to a 28x28 grayscale image.

    Parameters:
    grayscale_array (numpy.ndarray): A 1D array of shape (784,) containing grayscale values (0-255).

    Returns:
    None: Displays the image using matplotlib.
    """
    # Ensure the input is a NumPy array
    grayscale_array = np.array(grayscale_array)

    # Check if the input array has the correct shape
    if grayscale_array.shape != (784,):
        raise ValueError("Input array must have shape (784,)")

    # Check if the input array contains values between 0 and 255
    if np.min(grayscale_array) < 0 or np.max(grayscale_array) > 255:
        raise ValueError("Input array must contain values between 0 and 255")

    # Reshape the array to 28x28
    image = grayscale_array.reshape(28, 28)

    # Create a new figure
    plt.figure(figsize=(5, 5))

    # Display the image
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Turn off axis labels
    plt.title('28x28 Grayscale Image')

    # Show the plot
    plt.show()

# Example usage:
# Assuming you have a grayscale NumPy array called 'my_array' of shape (784,)
# grayscale_array_to_image(my_array)

def array_to_image(input_array, binarize=False, threshold=127):
    """
    Convert a NumPy array of shape (784,) to a 28x28 image.
    Can handle binary, grayscale, or binarized grayscale inputs.

    Parameters:
    input_array (numpy.ndarray): A 1D array of shape (784,) containing values (0-255 for grayscale, or 0-1 for binary).
    binarize (bool): If True, binarize the input array. Default is False.
    threshold (int): Threshold for binarization. Default is 127.

    Returns:
    None: Displays the image using matplotlib.
    """
    # Ensure the input is a NumPy array
    input_array = np.array(input_array)

    # Check if the input array has the correct shape
    if input_array.shape != (784,):
        raise ValueError("Input array must have shape (784,)")

    # Determine if the input is binary or grayscale
    is_binary = np.all(np.isin(input_array, [0, 1]))

    if is_binary and not binarize:
        # Binary input
        image = input_array.reshape(28, 28)
        cmap = 'binary'
        title = '28x28 Binary Image'
        vmin, vmax = 0, 1
    else:
        # Grayscale input
        if np.min(input_array) < 0 or np.max(input_array) > 255:
            raise ValueError("Grayscale values must be between 0 and 255")

        if binarize:
            # Binarize the grayscale input
            image = (input_array > threshold).astype(int).reshape(28, 28)
            cmap = 'binary'
            title = '28x28 Binarized Image'
            vmin, vmax = 0, 1
        else:
            # Keep as grayscale
            image = input_array.reshape(28, 28)
            cmap = 'gray'
            title = '28x28 Grayscale Image'
            vmin, vmax = 0, 255

    # Create a new figure
    plt.figure(figsize=(5, 5))

    # Display the image
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')  # Turn off axis labels
    plt.title(title)

    # Show the plot
    plt.show()

# Example usage:
# Assuming you have a NumPy array called 'my_array' of shape (784,)
# array_to_image(my_array)  # For grayscale or binary input
# array_to_image(my_array, binarize=True)  # To binarize grayscale input
# array_to_image(my_array, binarize=True, threshold=100)  # Custom threshold

def plot_time_sequence(time_sequence, title="Time Sequence Plot", x_label="Time", y_label="Value"):
    """
    Plot a line graph of a time sequence.

    Parameters:
    time_sequence (numpy.ndarray): A 1D NumPy array containing the time sequence data.
    title (str): The title of the plot. Default is "Time Sequence Plot".
    x_label (str): The label for the x-axis. Default is "Time".
    y_label (str): The label for the y-axis. Default is "Value".

    Returns:
    None: Displays the plot using matplotlib.
    """
    # Ensure the input is a NumPy array
    time_sequence = np.array(time_sequence)

    # Check if the input is a 1D array
    if time_sequence.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array")

    # Create x-axis values (assuming equal time intervals)
    x = np.arange(len(time_sequence))

    # Create the plot
    plt.figure(figsize=(16, 3))
    plt.plot(x, time_sequence, linewidth=2, color='blue')

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add markers for data points
    plt.scatter(x, time_sequence, color='red', s=30)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming you have a NumPy array called 'my_time_sequence'
# plot_time_sequence(my_time_sequence)

def plot_time_sequences(time_sequences, labels=None, single_chart=True, title="Time Sequence Plot", x_label="Time", y_label="Value"):
    """
    Plot line graphs of multiple time sequences.

    Parameters:
    time_sequences (list): A list of 1D NumPy arrays, each containing a time sequence.
    labels (list): A list of labels for each time sequence. If None, sequences will be numbered.
    single_chart (bool): If True, plot all sequences on one chart. If False, create separate charts.
    title (str): The title of the plot(s). Default is "Time Sequence Plot".
    x_label (str): The label for the x-axis. Default is "Time".
    y_label (str): The label for the y-axis. Default is "Value".

    Returns:
    None: Displays the plot(s) using matplotlib.
    """
    # Ensure all inputs are NumPy arrays
    time_sequences = [np.array(seq) for seq in time_sequences]

    # Check if all inputs are 1D arrays
    if any(seq.ndim != 1 for seq in time_sequences):
        raise ValueError("All inputs must be 1D NumPy arrays")

    # Create labels if not provided
    if labels is None:
        labels = [f"Sequence {i+1}" for i in range(len(time_sequences))]
    elif len(labels) != len(time_sequences):
        raise ValueError("Number of labels must match number of sequences")

    # Set up colors for multiple sequences
    colors = plt.cm.rainbow(np.linspace(0, 1, len(time_sequences)))

    if single_chart:
        # Plot all sequences on one chart
        plt.figure(figsize=(12, 6))
        for seq, label, color in zip(time_sequences, labels, colors):
            x = np.arange(len(seq))
            plt.plot(x, seq, linewidth=2, label=label, color=color)
            plt.scatter(x, seq, s=30, color=color)

        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        # Plot each sequence on a separate chart
        num_sequences = len(time_sequences)
        fig, axes = plt.subplots(num_sequences, 1, figsize=(12, 5*num_sequences), sharex=True)
        fig.suptitle(title, fontsize=16)

        for i, (seq, label, color) in enumerate(zip(time_sequences, labels, colors)):
            ax = axes[i] if num_sequences > 1 else axes
            x = np.arange(len(seq))
            ax.plot(x, seq, linewidth=2, label=label, color=color)
            ax.scatter(x, seq, s=30, color=color)
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        axes[-1].set_xlabel(x_label, fontsize=12)
        plt.tight_layout()
        plt.show()

# Example usage:
# time_seq1 = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
# time_seq2 = np.cos

def normalize_time_series(data):
    """
    Normalize a time series data around its mean.

    Args:
    data (list or numpy.array): The input time series data.

    Returns:
    numpy.array: The normalized time series data.
    """
    # Convert input to numpy array if it's not already
    data = np.array(data)

    # Calculate the mean of the data
    mean = np.mean(data)

    # Subtract the mean from each data point
    normalized_data = data - mean

    return normalized_data

def apply_notch_filter(eeg_data, fs, notch_freqs, notch_widths):
    """
    Apply a notch filter to EEG data to remove line noise.

    Args:
        eeg_data (numpy.ndarray): EEG data (channels x samples)
        fs (float): Sampling frequency of the EEG data
        notch_freqs (list or numpy.ndarray): List of notch frequencies (in Hz)
        notch_widths (list or numpy.ndarray): List of notch widths (in Hz)

    Returns:
        numpy.ndarray: Filtered EEG data
    """
    # Check if input data has the correct format (channels, samples)
    if not isinstance(eeg_data,pd.Series) and eeg_data.ndim == 1:
        eeg_data = eeg_data.to_numpy().reshape(1, -1)
    elif isinstance(eeg_data,pd.Series):
        if isinstance(eeg_data.iloc[0],list): #added iloc to remove warning messages
            eeg_data = np.array([item[0] for item in eeg_data])


    filtered_data = eeg_data.copy()

    for freq, width in zip(notch_freqs, notch_widths):
        # Design notch filter
        notch_filter = iirnotch(freq, width, fs)

        # Apply notch filter to each channel
        for channel in filtered_data.keys():
            filtered_data[channel] = filtfilt(notch_filter[0], notch_filter[1], filtered_data[channel])

    return filtered_data

def apply_dc_filter(eeg_data, fs, cutoff, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq

    # Compute Butterworth filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # Check if input data has the correct format (channels, samples)
    if not isinstance(eeg_data,pd.Series) and eeg_data.ndim == 1:
        eeg_data = eeg_data.to_numpy().reshape(1, -1)
    elif isinstance(eeg_data,pd.Series):
        if isinstance(eeg_data[0],list):
            eeg_data = np.array([item[0] for item in eeg_data])

    filtered_data = eeg_data.copy()
    # Apply the filter to each channel
    #filtered_data = np.zeros_like(data)
    for channel in filtered_data.keys():
        filtered_data[channel] = filtfilt(b, a, filtered_data[channel])

    return filtered_data


def apply_butterworth_filter(eeg_data, fs, lowcut, highcut, order=6):
    """
    Apply a non-causal Butterworth bandpass filter to the input data.

    Args:
        data (numpy.ndarray): Input data (channels x samples)
        fs (float): Sampling frequency of the data
        lowcut (float): Low-cutoff frequency (Hz)
        highcut (float): High-cutoff frequency (Hz)
        order (int): Order of the Butterworth filter (default: 6)

    Returns:
        numpy.ndarray: Filtered data
    """
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    # Compute Butterworth filter coefficients
    b, a = butter(order, [low, high], btype='band', analog=False)
    # Check if input data has the correct format (channels, samples)
    if not isinstance(eeg_data,pd.Series) and eeg_data.ndim == 1:
        eeg_data = eeg_data.to_numpy().reshape(1, -1)
    elif isinstance(eeg_data,pd.Series):
        if isinstance(eeg_data.iloc[0],list): #added iloc to remove warning messages
            eeg_data = np.array([item[0] for item in eeg_data])

    filtered_data = eeg_data.copy()
    # Apply the filter to each channel
    #filtered_data = np.zeros_like(data)
    for channel in filtered_data.keys():
        filtered_data[channel] = filtfilt(b, a, filtered_data[channel])

    return filtered_data

def fir_bandpass(eeg_data, fs, lowcut, highcut):
    numtaps = 1057
    cutoff = [lowcut, highcut]
    fir_coeffients = firwin(numtaps, cutoff, window='hamming', pass_zero=False, fs=fs)
    # Check if input data has the correct format (channels, samples)
    if not isinstance(eeg_data,pd.Series) and eeg_data.ndim == 1:
        eeg_data = eeg_data.to_numpy().reshape(1, -1)
    elif isinstance(eeg_data,pd.Series):
        if isinstance(eeg_data[0],list):
            eeg_data = np.array([item[0] for item in eeg_data])

    filtered_data = eeg_data.copy()
    # Apply the filter to each channel
    #filtered_data = np.zeros_like(data)
    for channel in filtered_data.keys():
        filtered_data[channel] = lfilter(fir_coeffients, 1.0, filtered_data[channel])

    return filtered_data

def read_dat_file(file_path, delimiter=';',type='float32'):
    """
    Reads a row-separated data file into a NumPy array.

    Args:
        file_path (str): Path to the data file.
        delimiter (str, optional): Delimiter used to separate values in each row. Default is ';'.

    Returns:
        numpy.ndarray: NumPy array containing the data from the file.
    """
    # with open(file_path, 'r') as file:
    #     data = []
    #     for line in file:
    #         row = line.strip().split(delimiter)
    #         data.append(row)
    arr = np.genfromtxt(file_path,dtype=type)

    return arr

def write_array_to_dat_file(array, filename):
    """
    Write a 1D NumPy array to a file, with each item on a new line.

    Args:
    array (numpy.ndarray): The 1D NumPy array to write.
    filename (str): The name of the file to write to.
    """
    # Ensure the array is 1D
    if array.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array")

    # Open the file in write mode
    with open(filename, 'w') as file:
        # Iterate through the array and write each item to a new line
        for item in array:
            file.write(f"{item}\n")

def filter_keys_and_label(data_list, keys_to_import, label):
    """
    Filters specific keys and a label from a list of dictionaries and returns a list of dictionaries.

    Parameters:
    - data_list (list): List of dictionaries containing the data.
    - keys_to_import (list): List of keys to be included in the output dictionaries.
    - label (str): The key for the label to be included in the output dictionaries.

    Returns:
    - list: A list of dictionaries with only the specified keys and the label key.

    Example:
    >>> data_list = [
    >>>     {'EEGdata_TP9': 1, 'EEGdata_TP10': 2, 'EEGdata_AF7': 3, 'EEGdata_AF8': 4, 'digit_label': 5},
    >>>     {'EEGdata_TP9': 6, 'EEGdata_TP10': 7, 'EEGdata_AF7': 8, 'EEGdata_AF8': 9, 'digit_label': 10}
    >>> ]
    >>> keys_to_import = ['EEGdata_TP9', 'EEGdata_TP10', 'EEGdata_AF7', 'EEGdata_AF8']
    >>> label = 'digit_label'
    >>> filtered_list = filter_keys_and_label(data_list, keys_to_import, label)
    >>> print(filtered_list)
    [
        {'EEGdata_TP9': [1], 'EEGdata_TP10': [2], 'EEGdata_AF7': [3], 'EEGdata_AF8': [4], 'digit_label': 5},
        {'EEGdata_TP9': [6], 'EEGdata_TP10': [7], 'EEGdata_AF7': [8], 'EEGdata_AF8': [9], 'digit_label': 10}
    ]
    """
    series_list = []
    for data in data_list:
        signals = {key: [value] for key, value in data.items() if key in keys_to_import}
        signals[label] = data[label]
        series_list.append(signals)
    return series_list

# Example usage

    # data_list = [
    #     {'EEGdata_TP9': 1, 'EEGdata_TP10': 2, 'EEGdata_AF7': 3, 'EEGdata_AF8': 4, 'digit_label': 5},
    #     {'EEGdata_TP9': 6, 'EEGdata_TP10': 7, 'EEGdata_AF7': 8, 'EEGdata_AF8': 9, 'digit_label': 10}
    # ]
    # keys_to_import = ['EEGdata_TP9', 'EEGdata_TP10', 'EEGdata_AF7', 'EEGdata_AF8']
    # label = 'digit_label'
    # filtered_list = filter_keys_and_label(data_list, keys_to_import, label)
    # print(filtered_list)

def plot_4_signals(df_record, title="Four Signal Plots", x_label="Time", y_labels=None, digit_label=None, norm=False, fs=None, x_units=None, y_units=None):
    """
    Plot 4 time series signals from a DataFrame record in a 4x1 figure.

    Parameters:
    - df_record: pandas Series or DataFrame row containing 4 numpy arrays
    - title: str, main title for the entire figure
    - x_label: str, label for the common x-axis
    - y_labels: list of str, labels for each y-axis (should contain 4 labels)

    Returns:
    - fig: matplotlib figure object
    """

    # Ensure we have 4 signals + a label
    if len(df_record) != 5:
        raise ValueError("Input should contain exactly 4 time series arrays and a label")

    # Create default y-labels if not provided
    if y_labels is None:
        y_labels = [f"Signal {i+1}" for i in range(4)]
    elif len(y_labels) != 4:
        raise ValueError("y_labels should contain exactly 4 labels")

    if digit_label is None:
        digit_label = -3   ## unknown label

    # Create the figure and subplots
    fig, axs = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Plot each signal
    for i, (signal_name, signal_data) in enumerate(df_record[y_labels].items()):
        norm_txt = ""
        if isinstance(signal_data,list) and len(signal_data) == 1:
            signal_data = signal_data[0]
        elif isinstance(signal_data,np.ndarray):
            pass
        else:
            raise TypeError("signal data should be a list of numpy array or numpy array")
        if norm:
            signal_data = normalize_time_series(signal_data)
            norm_txt = "Normalized"
        if x_label.lower() == 'time' and fs:
            time = np.arange(signal_data.size) / fs
            axs[i].plot(time, signal_data)
        else:
            axs[i].plot(signal_data)

        axs[i].set_ylabel(f"{y_labels[i]} {y_units}", fontsize = 10)
        axs[i].grid(True)
        axs[i].set_title(f"{signal_name} {norm_txt}", fontsize=12)

    # Set common x-label
    axs[-1].set_xlabel(f"{x_label} {x_units}", fontsize = 10)

    # Adjust layout
    plt.tight_layout()

    return fig


def band_frequancy_intersections(freqs,bands):
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= bands["Delta"]["low"], freqs <= bands["Delta"]["high"])
    idx_theta = np.logical_and(freqs >= bands["Theta"]["low"], freqs <= bands["Theta"]["high"])
    idx_alpha = np.logical_and(freqs >= bands["Alpha"]["low"], freqs <= bands["Alpha"]["high"])
    idx_beta = np.logical_and(freqs >= bands["Beta"]["low"], freqs <= bands["Beta"]["high"])
    idx_gamma = np.logical_and(freqs >= bands["Gamma"]["low"], freqs <= bands["Gamma"]["high"])

    return idx_delta,idx_theta,idx_alpha,idx_beta,idx_gamma

# function to Create windowed data based on window 32, overlap 4 = step size 28
def sliding_window_eeg(signal, window_size=32, overlap=4):
    """
    Apply a sliding window with overlap to a 2-second EEG signal.

    Parameters:
    signal (numpy.ndarray): 1D array of EEG signal data (256 samples)
    window_size (int): Size of each window (default: 32)
    overlap (int): Number of overlapping samples between windows (default: 4)

    Returns:
    numpy.ndarray: 2D array of windowed data
    """
    if len(signal) != 256:
        raise ValueError("Signal length must be 256 samples (2 seconds at 128Hz)")

    # Calculate the step size
    step = window_size - overlap

    # Calculate the number of windows
    num_windows = (len(signal) - window_size) // step + 1

    # Create an empty array to store the windowed data
    windowed_data = np.zeros((num_windows, window_size, 1))

    # Apply the sliding window
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        windowed_data[i] = signal[start:end].reshape(window_size,1)
    return windowed_data

# ## Create spectrogram image
# def sliding_window_eeg(signal, window_size=32, overlap=4):
#     """
#     Apply a sliding window with overlap to a 2-second EEG signal.

#     Parameters:
#     signal (numpy.ndarray): 1D array of EEG signal data (256 samples)
#     window_size (int): Size of each window (default: 32)
#     overlap (int): Number of overlapping samples between windows (default: 4)

#     Returns:
#     numpy.ndarray: 2D array of windowed data
#     """
#     if len(signal) != 256:
#         raise ValueError("Signal length must be 256 samples (2 seconds at 128Hz)")

#     # Calculate the step size
#     step = window_size - overlap

#     # Calculate the number of windows
#     num_windows = (len(signal) - window_size) // step + 1

#     # Create an empty array to store the windowed data
#     windowed_data = np.zeros((num_windows, window_size, 1))

#     # Apply the sliding window
#     for i in range(num_windows):
#         start = i * step
#         end = start + window_size
#         windowed_data[i] = signal[start:end].reshape(window_size,1)
#     return windowed_data


# w_data = sliding_window_eeg(df[df[label]==3].iloc[4]['EEGdata_AF3'],16,2)
# w_data.shape
