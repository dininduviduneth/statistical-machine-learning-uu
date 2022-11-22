import matplotlib.pyplot as plt
import numpy as np

def draw_linegraph(data_x, data_y):
    plt.plot(data_x, data_y)

def draw_histogram(data, bins, xlabel, ylabel, title, plot_mean = False):
    """
    Draws a basic histogram
    Arguments:
        data: an array of datapoints
        bins: number of bins
        xlabel: label for X-axis
        ylabel: label for Y-axis
        title: title of the histogram
    Returns:
        No return - only plots the histogram
    """

    plt.hist(data, bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if plot_mean:
        # Plot the mean line
        plt.axvline(data.mean(), color='k', linestyle='dashed', linewidth=1)

        # Print the mean value in histogram
        min_ylim, max_ylim = plt.ylim()
        plt.text(data.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(data.mean()))   
    else:
        pass

    plt.show()

def plot_scatter(data_x, data_y, xlabel, ylabel, title, plot_line = False):
        
    if plot_line:
        #find line of best fit
        a, b = np.polyfit(data_x, data_y, 1)

        #add points to plot
        plt.scatter(data_x, data_y)

        #add line of best fit to plot
        plt.plot(data_x, a*data_x+b, color='red')
    else:
        pass

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()