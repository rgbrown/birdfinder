# Rotokare Acoustic Monitoring

A Python framework for acoustic location and imaging of bird calls using a distributed microphone array.

This project uses **Generalized Cross-Correlation with Phase Transform (GCC-PHAT)** to localize sound sources in 3D space from multi-channel audio recordings. It was designed for monitoring native bird species (e.g., Kiwi, Tūī) at Rotokare Scenic Reserve.

## Project Structure

- `src/`: Core library for audio processing and geometry calculations.
- `notebooks/`: Jupyter notebooks for demonstrations and analysis.
- `data/`: Folder for microphone coordinates and audio files.
- `main.py`: CLI script for processing batches of audio files.
- `demo` : 

## Getting Started

### Prerequisites

- Python (developed and tested on python 3.14)
- packages in requirements.txt

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rgbrown/birdfinder.git
   cd birdfinder
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Put demo audio file [tenmin.wav](https://masseyuni-my.sharepoint.com/:u:/g/personal/rgbrown_massey_ac_nz/IQDEdHxqzyXKSbnAeiEsgN_lATY8sB-_kL3Xq1GnY4VGzak?e=dMWzzo) into the `data/raw` folder (too big for repository)

## Run the demo

The demo audio file `tenmin.wav` already has the first two channels removed. For running on an arbitrary MixPre recording, use the flag `--channels 2 3 4 5 6 7 8 9`

```bash
python main.py demo/files_to_process.csv --channels 0 1 2 3 4 5 6 7
```
