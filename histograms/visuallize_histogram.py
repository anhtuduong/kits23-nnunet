import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
import os

class HistogramVisualizer:
    def __init__(self, histograms_path):
        self.histograms = np.load(histograms_path, allow_pickle=True).item()
        self.gaussian_fits = {}
        self.labels_of_interest = {
            1: 'Kidney',
            2: 'Tumor',
            3: 'Cyst'
        }

    def compute_gaussian_fits(self):
        for label, histogram in self.histograms.items():
            print(f"Computing Gaussian fit for label: {label}")
            # Assume the bins are equally spaced and generate bin centers
            bin_centers = np.arange(len(histogram))

            try:
                # Fit a Gaussian distribution to the histogram
                popt, _ = curve_fit(self.gaussian, bin_centers, histogram, p0=[np.max(histogram), np.mean(bin_centers), np.std(bin_centers)])

                # Extract fitted params (amplitude, mean, std)
                amplitude, mean, std = popt
                self.gaussian_fits[label] = {'amplitude': amplitude, 'mean': mean, 'std': std}
                print(f"Fit parameters for {label}: amplitude={amplitude}, mean={mean}, std={std}")
            except RuntimeError:
                print(f"Warning: Gaussian fit did not converge for label {label}")
                self.gaussian_fits[label] = {'amplitude': np.max(histogram), 'mean': np.mean(bin_centers), 'std': np.std(bin_centers)}  # Use default values if fit fails

    def gaussian(self, x, amplitude, mean, std):
        return amplitude * norm.pdf(x, mean, std)

    def plot_histograms_with_fits(self):
        for label, histogram in self.histograms.items():
            label_name = self.labels_of_interest.get(label, f'Label {label}')
            print(f"Plotting histogram for label: {label_name}")
            bin_centers = np.arange(len(histogram))
            plt.figure(figsize=(10, 6))
            plt.bar(bin_centers, histogram, width=1.0, color='blue', alpha=0.7, label='Histogram')

            if label in self.gaussian_fits:
                params = self.gaussian_fits[label]
                fitted_gaussian = self.gaussian(bin_centers, params['amplitude'], params['mean'], params['std'])
                plt.plot(bin_centers, fitted_gaussian, color='red', linestyle='dashed', linewidth=2, label='Fitted Gaussian')

            plt.xlabel('Bins')
            plt.ylabel('Counts')
            plt.title(f'Histogram and Gaussian Fit for {label_name}')
            plt.legend()

            # Save the plot as an image file
            plt.savefig(f'hists/histogram_fit_{label_name.lower()}.png')
            plt.close()

# Example usage
histograms_path = 'hists/histogram_counts.npy'  # Replace with your histograms .npy file path

# Ensure the provided file path is correct and exists
if os.path.isfile(histograms_path):
    visualizer = HistogramVisualizer(histograms_path)
    visualizer.compute_gaussian_fits()
    visualizer.plot_histograms_with_fits()
else:
    print("Please provide a valid path for the histogram counts .npy file.")
