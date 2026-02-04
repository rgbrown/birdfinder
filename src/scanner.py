import numpy as np
import pandas as pd
from tqdm import tqdm # for progress bar
from src.micarray import MicArray

class BirdScanner:
    def __init__(self, mic_array: MicArray):
        self.array = mic_array

    def _process_window(self, recording, t_start, window_dur, X, Y, z_source, 
                        threshold, tolerance, mask_size, pairs, max_peaks, 
                        envelope, filter, low_cut, high_cut):
        
        """
        Process a single time window. 
        Returns a list of detection dictionaries (empty if none found)
        """

        window = recording.get_window(t_start, window_dur)

        gcc_data = self.array.cache_gcc_data(window, 
                                             beta=0.6, 
                                             envelope=envelope, 
                                             filter=filter, 
                                             low_cut=low_cut, 
                                             high_cut=high_cut, 
                                             pairs=pairs)

                    # How big is the biggest peak?
        peak_strength = np.max([np.max(c) for c in gcc_data['correlations']])
        
        # Return early if none big enough
        if peak_strength < threshold:
            return []
            
        # Compute Heatmap
        heatmap = self.array.generate_heatmap(gcc_data, X, Y, z_source=z_source)
        
        # Find Peaks
        peaks = self.array.find_peaks(heatmap, X, Y, 
                                      tolerance=tolerance, 
                                      mask_size=mask_size, 
                                      max_peaks=max_peaks)

        detections = []
        for p in peaks:
            detections.append({
                'time_start': t_start,
                'duration': window_dur,
                'x': p['x'],
                'y': p['y'],
                'z': z_source,
                'strength': p['score']
            })

        return detections
        
    def scan_file(self, recording, X, Y, stride=0.25, window_dur=0.5, df_file=None,
                  threshold=1e-6, z_source=0.0, tolerance=0.7, mask_size=10,
                  pairs=None, max_peaks=2, envelope=True, filter=True, low_cut=100, high_cut=10000):
        """
        Scans a recording for sources.
        
        Args:
            stride: Step size in seconds.
            threshold: Min GCC strength to trigger a detection. Highly parameter and data dependent
            tolerance, mask_size, max_peaks: passed through to find_peaks
            z_source: Assumed height of birds
        """
        
        detections = []
        
        # Define Windows
        duration = recording.frames / recording.fs
        possible_starts = np.arange(0, duration - window_dur, stride)

        # Iteratively remove possible starts not in the right time frame
        if df_file is None:
            starts = possible_starts
        else:
            starts = []
            for s in possible_starts:
                for i in range(len(df_file)):
                    if (s >= df_file.iloc[i]['start']) & (s <= df_file.iloc[i]['end']):
                        starts.append(s)
            starts = np.array(starts)
        
        print(f"Scanning {duration:.1f}s of audio ({len(starts)} windows)...")
        
        for t_start in tqdm(starts):
            new_detections = self._process_window(
                recording=recording,
                t_start=t_start,
                window_dur=window_dur,
                X=X, Y=Y,
                z_source=z_source,
                threshold=threshold,
                tolerance=tolerance,
                mask_size=mask_size,
                pairs=pairs,
                max_peaks=max_peaks,
                envelope=envelope,
                filter=filter,
                low_cut=low_cut,
                high_cut=high_cut
            )
            detections.extend(new_detections)
                
        return pd.DataFrame(detections)
    
    def refine_with_height(self,detections,recording,window_dur,search_radius,pairs,threshold):
        x_loc = np.linspace(-search_radius, search_radius, 100)
        y_loc = np.linspace(-search_radius, search_radius, 100)
        X_loc, Y_loc = np.meshgrid(x_loc, y_loc)
        Z = np.arange(0, 7, 0.5)
        refined = []
        for index, row in tqdm(detections.iterrows()):
            x0 = row['x']
            y0 = row['y']
            window = recording.get_window(row['time_start'], window_dur)
            gcc_data = self.array.cache_gcc_data(window, 
                                            pairs=pairs, 
                                            beta=0.6, # GCC-PHAT tunable parameter, calculates R / np.abs(R)**beta, makes it work better
                                            envelope=True, # Hilbert transform to vaguely follow the GCC envelope, helpful for narrow band
                                            filter=True, # Band pass filter the signal
                                            low_cut=100, 
                                            high_cut=10000)
            px, py, pz, strength = self.array.refine_height(gcc_data, X_loc + x0, Y_loc + y0, Z, mask_size=20)
            if strength>threshold:
                refined.append({
                    'time_start': row['time_start'],
                    'duration': window_dur,
                    'x': px,
                    'y': py,
                    'z': pz,
                    'strength':strength,
                    't_m': int(row['time_start']/60),
                    't_s': row['time_start'] - int(row['time_start']/60)*60,
                    'species':''
                })
        return pd.DataFrame(refined)
    
def merge_detections(df,max_separation_time=1,max_separation_space=1):
    # Combine detections that are close together in time and space
    # There is probably a nice way to do this, this isn't it
    ind = 0
    newdf = df.iloc[0:1][:].copy()
    df['used'] = False

    while ind<len(df)-1:
        x = df.iloc[ind]['x']
        y = df.iloc[ind]['y']
        z = df.iloc[ind]['z']
        t = df.iloc[ind]['time_start']
        dt = df.iloc[ind]['duration']

        df.loc[ind,'used']=True

        df['distance'] = np.sqrt((df['x']-x)**2+(df['y']-y)**2+(df['z']-z)**2)
        df['timesep'] = np.abs(df['time_start'] - t - dt)
        close = df.index[(df['distance']<max_separation_space) & (df['timesep']<=max_separation_time) & (~df['used'])]
        if len(close)>0:
            df.loc[close,'used']=True
            # Make the duration be the longest time
            newdf.loc[len(newdf)-1,'duration'] = df.loc[close[-1],'time_start']-t+df.loc[close[-1],'duration']
        while (df.loc[ind]['used']):
            if ind < len(df)-1:
                ind+=1
            else:
                break
        newdf = pd.concat([newdf,df.loc[ind:ind,['time_start','duration','x','y','z','strength','t_m','t_s','species']]],ignore_index=True)
    return newdf


