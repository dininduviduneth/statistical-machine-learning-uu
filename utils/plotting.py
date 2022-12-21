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
        plot_mean: print the mean or not? defaults to False
    Returns:
        No return - only plots the histogram
    """

    plt.hist(data, bins, color = "dodgerblue")
    plt.rcParams.update({'font.size': 15})
    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel(ylabel, fontsize=17)
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
    
def draw_double_histogram(data_1, data_2, bins, data_1_legend, data_2_legend, xlabel, ylabel, title, show_legend = False, plot_mean = False):
    from matplotlib import pyplot

    pyplot.hist(data_1, bins, alpha=0.5, label=data_1_legend, color = "dodgerblue")
    pyplot.hist(data_2, bins, alpha=0.3, label=data_2_legend, color = "red")
    plt.rcParams.update({'font.size': 12})
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.title(title)

    if show_legend:
        pyplot.legend(loc='center right', bbox_to_anchor=(1.30, 0.5))
    
    pyplot.show()

def plot_scatter(data_x, data_y, xlabel, ylabel, title, plot_line = False):
    """
    Plots a basic scatter with the line of best fit
    Arguments:
        data_x: an array of datapoints for X-axis
        data_y: an array of datapoints for Y-axis
        xlabel: label for X-axis
        ylabel: label for Y-axis
        title: title of the Scatter Plot
        plot_line: Plot line of best fit or not - defaults to False
    Returns:
        No return - only plots the scatter plot
    """

    if plot_line:
        #find line of best fit
        a, b = np.polyfit(data_x, data_y, 1)

        #add points to plot
        plt.scatter(data_x, data_y, c = "dodgerblue", alpha = 1)

        #add line of best fit to plot
        plt.plot(data_x, a*data_x+b, color='red')
    else:
        #add points to plot
        plt.scatter(data_x, data_y, c = "dodgerblue", alpha = 1)

    plt.rcParams.update({'font.size': 15})
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.grid(alpha = 0.5)
    plt.title(title)

    plt.show()