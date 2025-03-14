import mne
import matplotlib.pyplot as plt

path = "raw_dataset/eeg_dataset/sub-01/eeg/sub-01_task-rsvp_eeg.vhdr"
raw = mne.io.read_raw_brainvision(path, preload=True)
# Check basic info
print(raw.info)

# Plot the raw EEG signal
raw.plot(n_channels = 5, scalings = "auto")
plt.show()

raw.filter(l_freq=0.4, h_freq=60)
raw.plot(n_channels = 5, scalings = "auto")
plt.show()