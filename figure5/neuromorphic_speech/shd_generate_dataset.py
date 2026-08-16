# copy for https://github.com/byin-cwi/Efficient-spiking-networks.git
import argparse
from pathlib import Path
# from keras.utils import get_file
"""
The dataset is 48kHZ with 24bits precision
* 700 channels
* longest 1.17s
* shortest 0.316s
"""

# cache_dir=os.path.expanduser("~/data")
# cache_subdir="hdspikes"
# print("Using cache dir: %s"%cache_dir)
#
# # The remote directory with the data files
# base_url = "https://compneuro.net/datasets"

# Retrieve MD5 hashes from remote



# file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }

import tables
import numpy as np


def binary_image_readout(times,units,dt = 1e-3):
    img = []
    N = int(1/dt)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    return np.array(img)

def binary_image_spatical(times,units,dt = 1e-3,dc = 10):
    img = []
    N = int(1/dt)
    C = int(700/dc)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(C)# add spacial count
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    return np.array(img)


def generate_dataset(file_name,dt=1e-3):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ",len(times))
    X = []
    y = []
    for i in range(len(times)):
        tmp = binary_image_readout(times[i], units[i],dt=dt)
        X.append(tmp)
        y.append(labels[i])
    fileh.close()
    return np.array(X),np.array(y)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SHD HDF5 files to NumPy arrays.")
    parser.add_argument("--input-dir", type=Path, required=True, help="directory containing shd_{test,train}.h5")
    parser.add_argument("--output-dir", type=Path, required=True, help="destination directory")
    parser.add_argument("--dt", type=float, default=4e-3, help="time-bin width in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        source = args.input_dir / f"shd_{split}.h5"
        if not source.is_file():
            raise FileNotFoundError(f"Missing SHD source file: {source}")
        features, labels = generate_dataset(source, dt=args.dt)
        np.save(args.output_dir / f"{split}X_4ms.npy", features)
        np.save(args.output_dir / f"{split}Y_4ms.npy", labels)


if __name__ == "__main__":
    main()

