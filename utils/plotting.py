import matplotlib.pyplot as plt

def draw_histogram(data, bins, xlabel, ylabel, title):
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

    plt.show()