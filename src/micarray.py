import numpy as np
import scipy.ndimage
import scipy.signal
from dataclasses import dataclass
from src.audio import gcc_phat, AudioRecording
import matplotlib.pyplot as plt

@dataclass
class MicArray:
    mic_positions: np.ndarray # Shape (N, 3)
    fs: int
    temperature: float = 20.0 # just fudged for now to give 343, but could incorporate later

    @property
    def c(self):
        # temperature-dependent speed of sound
        return 331.3 + (0.606 * self.temperature)

    def predict_delays(self, source_pos):
        """ Calculate theoretical delays relative to center of array """

        # Calculate distance from source to all mics
        dists = np.linalg.norm(self.mic_positions - source_pos, axis=1)

        # Calculate time difference relative to (0, 0)
        tdoa = (dists - np.linalg.norm(source_pos)) / self.c

        return tdoa

    def compute_delay_grid(self, pair_indices, X_grid, Y_grid, z_source=0.0):
        """
        Calculates expected sample delay between two mics (i, j) for every point
        in the domain. A positive delay corresponds to microphone j being
        further away
        X_grid, Y_grid: the 2D mesh
        z_source: The assumed height of the source (default 0)
        """
        i, j = pair_indices
        p_i = self.mic_positions[i]
        p_j = self.mic_positions[j]

        # Distance to mic i
        dx_i = X_grid - p_i[0]
        dy_i = Y_grid - p_i[1]
        dz_i = z_source - p_i[2]
        dist_i = np.sqrt(dx_i**2 + dy_i**2 + dz_i**2)

        # Distance to mic j
        dx_j = X_grid - p_j[0]
        dy_j = Y_grid - p_j[1]
        dz_j = z_source - p_j[2]
        dist_j = np.sqrt(dx_j**2 + dy_j**2 + dz_j**2)

        # Delay in samples
        delay_seconds = (dist_j - dist_i) / self.c
        delay_samples = np.round(delay_seconds * self.fs).astype('int')
        
        return delay_samples

    def cache_gcc_data(self, audio_segment: AudioRecording, pairs=None, filter=False, 
                       low_cut=2000, high_cut=8000, beta=0.8, envelope=True):
        """ 
        Precompute GCC for specific pairs. Passes various filtering/tuning options to gcc-phat
        - bandpass filtering
        - Tunable PHAT
        - envelope
        """

        # If pairs not specified, just make a ring
        if pairs is None:
            n_mics = len(self.mic_positions)
            pairs = [(i, (i + 1) % n_mics) for i in range(n_mics)]

        # Only make a filter if asked for
        if filter:
            sos = scipy.signal.butter(4, [low_cut, high_cut], btype='band', fs=self.fs, output='sos')

        gcc_list = []


        for pair in pairs:
            raw1 = audio_segment.data[pair[0]]
            raw2 = audio_segment.data[pair[1]]

            if filter:
                audio1 = scipy.signal.sosfiltfilt(sos, raw1)
                audio2 = scipy.signal.sosfiltfilt(sos, raw2)
            else:
                audio1 = raw1
                audio2 = raw2

            corr, n_fft = gcc_phat(audio1, audio2, fs=self.fs, beta=beta)

            if envelope:
                corr = np.abs(scipy.signal.hilbert(corr))

            gcc_list.append(corr)

        return {
            'pairs': pairs, 
            'correlations': gcc_list,
            'fs': self.fs
        }

    def refine_height(self, gcc_data, X, Y, Z, mask_size=10):
        # Note: X, Y are localised around a previously found peak. We only want one
        if Z is None:
            Z = np.arange(0, 7, 0.5)

        P = []
        for z in Z:
            heatmap = self.generate_heatmap(gcc_data, X, Y, z_source=z)
            peaks = self.find_peaks(heatmap, X, Y, tolerance=0.95, mask_size=mask_size, max_peaks=1)
            P.append(peaks[0]['score'])

        z = Z[np.argmax(P)]
        heatmap = self.generate_heatmap(gcc_data, X, Y, z_source=z)
        peaks = self.find_peaks(heatmap, X, Y, tolerance=0.95, mask_size=mask_size, max_peaks=1)
        
        px = peaks[0]['x']
        py = peaks[0]['y']
        pz = z
        strength = peaks[0]['score']
        return px, py, pz, strength

    def generate_heatmap(self, gcc_data, X_grid, Y_grid, z_source=0.0, mask_level=10):
        """
        Adds gcc contribution from each pair to each grid point
        """

        nx, ny = X_grid.shape
        h_image = np.zeros((nx, ny))
        
        pairs = gcc_data['pairs']
        corrs = gcc_data['correlations']

        for idx, pair in enumerate(pairs):
            gcc = corrs[idx]

            # Get rid of negative correlations
            gcc_clean = np.maximum(gcc, 0)
            
            # Get expected delay at every pixel
            D_grid = self.compute_delay_grid(pair, X_grid, Y_grid, z_source=z_source)
            
            # Masking, to make strength die out as we get further away
            # (otherwise blurring gives wacky results)
            # Find center of this specific pair
            mic_center = np.mean([self.mic_positions[pair[0]], self.mic_positions[pair[1]]], axis=0)
            
            # Create mask
            h_mask = np.hypot(X_grid - mic_center[0], Y_grid - mic_center[1])
            max_mask = np.max(h_mask)
            if max_mask > 0:
                h_mask = mask_level * h_mask / max_mask
            
            h_mask[h_mask < 1] = 1
            
            # Add energy (using array indexing for the delay lookups)
            h_image += (gcc_clean[D_grid]**2) / h_mask

        return np.sqrt(h_image)
    
    def calibrate_interactively(self, gcc_data, X_grid, Y_grid, extent=None, z_source=0.0, db_scale=False, dynamic_range=60):
        """
        Opens a plot where mics can be dragged to recalibrate the array geometry.
        - Drag to move
        - Double click to reset a microphone position
        """
        if extent is None:
            extent = [X_grid.min(), X_grid.max(), Y_grid.min(), Y_grid.max()]

        # Store original positions
        original_positions = self.mic_positions.copy()

        def process_view(raw_heatmap):
            """
            Put it into dB form if required. Turns out this is a bad idea
            """
            if not db_scale:
                return raw_heatmap, None, None

            view_data = 20*np.log10(raw_heatmap + 1e-9)
            view_data = view_data - np.max(view_data)
            return view_data, -dynamic_range, 0
    
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initial Plot
        print(f"Computing initial heatmap... (source z={z_source} m)...")
        raw_map = self.generate_heatmap(gcc_data, X_grid, Y_grid, z_source=z_source) 
        view_map, vmin, vmax = process_view(raw_map)
        im = ax.imshow(view_map, origin='lower', extent=extent, 
                       cmap='inferno', vmin=vmin, vmax=vmax)
        
        # colorbar
        cbar_label = 'Strength (dB)' if db_scale else 'Strength (Linear)'
        plt.colorbar(im, label=cbar_label)

        # Plot mics  
        scat = ax.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                          c='cyan', s=150, edgecolors='white', picker=True, zorder=10)
        
        ax.set_title("Drag mics to move, double-click to reset mic")
        
        # State container
        state = {'ind': None}

        def on_pick(event):
            if event.artist != scat: return

            if event.mouseevent.dblclick:
                ind = event.ind[0]
                print(f"Resetting mic {ind} position ...")

                self.mic_positions[ind] = original_positions[ind]

                scat.set_offsets(self.mic_positions[:, :2])

                # Recompute heatmap 
                new_map = self.generate_heatmap(gcc_data, X_grid, Y_grid)
                view_map, vmin, vmax = process_view(new_map)
                im.set_data(new_map)
                im.set_clim(vmin=vmin, vmax=vmax)
                fig.canvas.draw_idle()
                
                # Stop the drag logic from engaging
                return

            state['ind'] = event.ind[0]

        def on_motion(event):
            if state['ind'] is None: return
            if event.inaxes != ax: return
            
            # Update Mic Position
            self.mic_positions[state['ind'], 0] = event.xdata
            self.mic_positions[state['ind'], 1] = event.ydata
            
            # Update Scatter Plot Visuals
            scat.set_offsets(self.mic_positions[:, :2])
            
            fig.canvas.draw_idle()

        def on_release(event):
            if state['ind'] is None: return
            state['ind'] = None
            
            # Recompute Heatmap with new positions
            print("Recalculating...")
            new_map = self.generate_heatmap(gcc_data, X_grid, Y_grid)
            view_map, vmin, vmax = process_view(new_map)
            
            # Update Image
            im.set_data(view_map)
            im.set_clim(vmin=vmin, vmax=vmax)
            fig.canvas.draw_idle()

        # Connect events
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        plt.show()

    def plot_heatmap(self, heatmap, extent=[-30, 30, -30, 30], db_scale=False, dynamic_range=60):
        fig, ax = plt.subplots()

        data = heatmap.copy()

        if db_scale:
            data = 20*np.log10(data + 1e-9)
            data = data - np.max(data)
            vmin = -dynamic_range
            vmax = 0
            label = 'Strength (dB relative to peak)'
        else:
            vmin = None
            vmax = None
            label = 'GCC combined strength (linear)'

        im = ax.imshow(data, 
                       origin="lower", 
                       extent=extent,
                       cmap='inferno',
                       interpolation='bicubic',
                       vmin=vmin,
                       vmax=vmax
            )
        plt.colorbar(im,label='GCC combined strength')
        plt.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                    c='white', marker='.', s=100, label="mics")
        return fig, ax

    

    def find_peaks(self, h_image, X_grid, Y_grid, tolerance=0.5, mask_size=10, blurring=2.0, max_peaks=3):
        """
        Finds local maxima in the heatmap.
        """
        # Blur
        img_blur = scipy.ndimage.gaussian_filter(h_image, sigma=blurring)
        
        # Peak finding stuffs
        max_val = np.max(img_blur)
        min_threshold = max_val * tolerance
        
        positions = []
        
        # Work on a copy so we don't destroy the original image
        search_img = img_blur.copy()
        
        # Safety break to prevent infinite loops
        for _ in range(max_peaks): 
            curr_max = np.max(search_img)
            if curr_max < min_threshold:
                break
                
            # Find location
            y_idxs, x_idxs = np.where(search_img == curr_max)
            y_idx, x_idx = y_idxs[0], x_idxs[0] # Take first match
            
            # Save physical location
            positions.append({
                'score': curr_max,
                'x': X_grid[y_idx, x_idx],
                'y': Y_grid[y_idx, x_idx]
            })
            
            # Zero out the neighborhood
            # (Simple rectangular mask)
            y_min = max(0, y_idx - mask_size)
            y_max = min(search_img.shape[0], y_idx + mask_size)
            x_min = max(0, x_idx - mask_size)
            x_max = min(search_img.shape[1], x_idx + mask_size)
            
            search_img[y_min:y_max, x_min:x_max] = 0
            
        return positions
    
def choose_pairs(audio,n_mics=8):
    # Decide which microphone pairs to use
    n = len(audio[:,0])
    powers = [np.sum(audio[:, i]**2) / len(audio[:,0]) for i in range(n_mics)]
    mics_sorted = np.argsort(powers)[::-1]
    
    match mics_sorted[0]:
        case 0:
            if powers[1] > powers[7]:
                pairs = [(0, 1), (0, 2), (7, 0), (7, 2)]
            else:
                pairs = [(7, 0), (6, 0), (6, 7)]
        case 1: 
            pairs = [(0, 1), (0, 2), (1, 4)]
        case 2:
            pairs = [(0, 2), (2, 4), (0, 4)]
        case 3: 
            pairs = [(3, 4), (2, 4), (2, 5)]
        case 4:
            if powers[3] > powers[5]:
                pairs = [(2, 4), (3, 4), (4, 5)]
            else:
                pairs = [(3, 4), (4, 5), (4, 6), (5, 6)]
        case 5:
            pairs = [(4, 5), (5, 6), (4, 6)]
        case 6:
            if powers[5] > powers[7]:
                pairs = [(4, 5), (5, 6), (4, 6), (6, 7)]
            else:
                pairs = [(5, 6), (6, 7), (7, 0), (6, 0)]
        case 7:
            pairs = [(6, 7), (7, 0), (6, 0)]
    return pairs
