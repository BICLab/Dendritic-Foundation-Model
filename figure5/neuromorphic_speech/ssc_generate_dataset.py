# copy for https://github.com/byin-cwi/Efficient-spiking-networks.git
import argparse
from pathlib import Path

"""
The dataset is 48kHZ with 24bits precision
* 700 channels
* longest 1.s
* shortest 0.21s
"""

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


def generate_dataset(file_name, output_dir, dt=1e-3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ",len(times))
    for i in range(len(times)):
        print(i)
        x_tmp = binary_image_readout(times[i], units[i],dt=dt)
        y_tmp = labels[i]
        output_file_name = output_dir / f"ID:{i}_{y_tmp}.npy"
        np.save(output_file_name, x_tmp)
    fileh.close()
    print('done..')
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SSC HDF5 files to NumPy samples.")
    parser.add_argument("--input-dir", type=Path, required=True, help="directory containing ssc_{test,valid,train}.h5")
    parser.add_argument("--output-dir", type=Path, required=True, help="destination root for split directories")
    parser.add_argument("--dt", type=float, default=4e-3, help="time-bin width in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    for split in ("test", "valid", "train"):
        source = args.input_dir / f"ssc_{split}.h5"
        if not source.is_file():
            raise FileNotFoundError(f"Missing SSC source file: {source}")
        generate_dataset(source, args.output_dir / split, dt=args.dt)


if __name__ == "__main__":
    main()
