#%%
import numpy as np
import matplotlib.pyplot as plt
from src.micarray import MicArray
from src.audio import AudioRecording
from src.scanner import BirdScanner, merge_detections
from pathlib import Path
import pandas as pd
import os
import argparse
from tqdm import tqdm

def quick_plot(mic_positions, df, name):
    # Quick Visualization of birds found
    plt.figure(figsize=(10, 10))

    # Plot Mics
    plt.scatter(mic_positions[:,0], mic_positions[:,1], c='red', marker='x', s=100, label='Mics')

    # Plot Detections
    # Colour by time to see movement patterns?
    if not df.empty:
        #print(10 + df['strength']*1e6)
        sc = plt.scatter(df['x'], df['y'], 
                    c=df['time_start'], cmap='viridis', 
                        s=10 + (df['strength'] * 1e7), # Size by loudness
                        alpha=0.6, label='Detections')
        plt.colorbar(sc, label='Time (s)')

    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Detected {len(df)} events")
    #plt.show()
    plt.savefig(f"{name}.png")

def start_process(csvfile, channels=None, datadir='data/raw'):
    # Parameters
    window_dur = 0.5 # s
    threshold = 1e-6 
    mask_radius = 2.0 # m (minimum distance for distinct detection)
    stride = 0.5 # s

    # Define grid (in m)
    xmin, xmax, ymin, ymax = [-30, 30, -30, 30]
    # How many x and y points in each grid direction
    nx = 1200
    ny = nx

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, nx)
    X, Y = np.meshgrid(x, y)

    mask_size = int(mask_radius / ((xmax - xmin)/nx))

    # Set up the microphone array
    mics = np.loadtxt("data/metadata/mic_coords.txt")
    array = MicArray(mics, fs=48000)
    scanner = BirdScanner(array) 

    # Use all pairs of microphones
    pairs = []
    for i in range(8):
        for j in range(i+1, 8):
            pairs.append((i, j))

    # Read a csv of files and timestamps (file, start, end) given as an input
    # Start and end are in seconds
    datadir = Path(datadir)
    timestamps = pd.read_csv(csvfile,names=['file','start','end'])
    list_files = pd.unique(timestamps['file'])
    print(f"{len(list_files)} files to process")
    for filename in list_files:
        name = os.path.split(filename)[-1].split('.')[0]
        fullpath = datadir / filename
        rec = AudioRecording(
            fullpath,
            lazy=True,
            channels=channels
            )
        
        print(f"Starting detection run on file {filename}...")

        # Timestamps for this file
        df_file = timestamps[timestamps['file']==filename]

        # Returns pandas dataframe
        unrefined = scanner.scan_file(rec, X, Y, 
                                    stride=stride, 
                                    window_dur=window_dur,
                                    df_file = df_file,
                                    threshold=threshold, 
                                    tolerance=0.95, 
                                    z_source=2.0, 
                                    pairs=pairs, 
                                    mask_size=mask_size, 
                                    max_peaks=2
                                    )

        # Save unrefined detections
        unrefined.to_csv(f"{name}_unrefined.csv", index=False)
        quick_plot(array.mic_positions,unrefined,name+'_unrefined')

        # Loop over each detection, find height, and refine position as side effect
        search_radius = 2
        refined = scanner.refine_with_height(unrefined,rec,window_dur,search_radius,pairs,threshold)
        refined.to_csv(f"{name}_refined.csv", index=False)
        lr = len(refined)+1
        while len(refined) < lr:
            lr = len(refined)
            refined = merge_detections(refined,max_separation_time=5,max_separation_space=4)
            #print(lr)
        refined.to_csv(f"{name}_refined_merged.csv", index=False)
        quick_plot(array.mic_positions,refined,name+'_refined_merged')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--channels', type=int, nargs='+', default=None, help='List of channels (e.g., --channels 2 3 4)')
    args = parser.parse_args()
    print(f"Channels: {args.channels}")

    start_process(args.filename, args.channels)
