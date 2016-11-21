import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_data_points(first_dimension, second_dimension, result_labels):     #result_labels = 0 or 1
    figure = plt.figure(figsize=(8,8))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(first_dimension, second_dimension, result_labels)
    plt.show()

def plot_results_points(evaluation_metrics_list, classifiers_list, result_matrix):     # (dimension -> classifiers X evaluation_metrics)
    figure_res, axes_res = plt.subplots()
    x_positions = np.arange(len(classifiers_list))
    width = 0.2
    axes_res.set_title('Comparison of evaluation_metrics for each classifier')

    for i in range(1,len(evaluation_metrics)+1):
        curr_plot = plt.subplot(len(evaluation_metrics), 1, i)
        curr_plot.bar(x_positions, result_matrix[i-1], width, alpha = 0.6, color='r')
        curr_plot.set_ylabel('Scores of classifiers')
        curr_plot.set_xticks(x_positions + width)
        curr_plot.set_xticklabels(classifiers_list)
    plt.show()
