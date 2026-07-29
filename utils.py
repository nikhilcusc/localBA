import matplotlib.pyplot as plt
import numpy as np
from scipy  import ndimage

def plot_slice(data, step=10, dim=3):
    plt.rcParams['figure.figsize'] = [50, 50]
    print(data.shape)
    last_slice = data.shape[dim] - (data.shape[dim]%step)
    fig, axs = plt.subplots(1, int(last_slice/step))
    if dim==3: 
        for s in range(0, last_slice, step):           
            temp = np.array(data[:, :, s])
            axs[int(s/step)].imshow(ndimage.rotate(temp, 270))
            axs[int(s/step)].axis('off')
    elif dim==2:
        for s in range(0, last_slice, step):
            temp = np.array(data[:, s, :])
            axs[int(s/step)].imshow(ndimage.rotate(temp, 270))
            axs[int(s/step)].axis('off')
    else:
        for s in range(0, last_slice, step):           
            temp = np.array(data[s, :, :])
            axs[int(s/step)].imshow(temp)
            axs[int(s/step)].axis('off')
    
    plt.show()
    
    return

def plot3Views(data, vmin=0, vmax=1, slice1=128, cmap='viridis', title='3D Views'):
    
    slice_front = data[:, :, slice1]  # middle slice along the z-axis
    slice_side = data[slice1, :, :]   # middle slice along the x-axis
    slice_top = data[:, slice1, :]    # middle slice along the y-axis

    if vmax==0:
        vmax = np.max(data)
    if vmin==0:
        vmin = np.min(data)
    # Create subplots for each view
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    im0 = axes[0].imshow(slice_front, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('Axial')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(slice_side, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('Sagittal')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(slice_top, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title('Coronal')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(title, fontsize=16)
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    return

def plotHist(data, title='Histogram of Non-Zero Elements in 3D Array'):
    # Flatten the 3D array into a 1D array
    flattened_array = data.flatten()

    # Remove zero values if needed
    flattened_array = flattened_array[flattened_array != 0]

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flattened_array, bins='auto', color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return

def scatterPlotCA(x, y, title="", xlabel="", ylabel="", color='blue', marker='o'):
    plt.scatter(x, y, label='Set 1', color=color, marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Calculate the line of best fit for the first set of points
    z1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(z1)
    plt.plot(x, p1(x), "k--", label="Line of Best Fit")
    line_of_best_fit_text_1 = f"y = {z1[0]:.2f}x + {z1[1]:.2f}"
    plt.text(min(x), min(y), line_of_best_fit_text_1, color='red', fontsize=10)

    # Add a legend
    plt.legend()
    plt.show()
    return None