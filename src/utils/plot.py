"""
Implement methods plotting and drawing figures.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com))
        Jianhai Su
"""
from enum import Enum
import numpy as np
from utils.csv_headers import IdealModelEvalHeaders as headers
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from utils.config import PATH, MODE

line_styles = ['-', '--', ':', '-.', '.']
line_width = 3.0
colors = ['magenta', 'deepskyblue', 'limegreen', 'red',
            'deeppink', 'darkorange', 'fuchsia', 'forestgreen', 'blue',
           'aqua', 'orangered', 'navy', 'darkgray', 'orange', 'black']
marks = ['o', 's', 'D', '+', '*', 'v', '^', '<', '>', '.', '+', 'p', 'h',  ',',
           'd', '|', '1', '2', '3', '4', '8', 'P', 'H', 'X', 'D']
nb_colors = len(colors)
nb_marks = len(marks)

class LEGEND_LOCATION(Enum):
    BEST = 'best'
    UPPER_RIGHT = 'upper right'
    UPPER_LEFT = 'upper left'
    LOWER_LEFT = 'lower left'
    LOWER_RIGHT = 'lower right'
    RIGHT = 'right'
    CENTER_LEFT = 'center left'
    CENTER_RIGHT = 'center right'
    LOWER_CENTER = 'lower center'
    UPPER_CENTER = 'upper center'
    CENTER = 'center'


class legend(object):
    def __init__(self):
        self.location = LEGEND_LOCATION.UPPER_CENTER.value
        self.ncol = 2
        self.box_anchor = None
        self.fontsize = 10
        self.fancybox = True
        self.shadow = True

    def set_location(self, location):
        self.location = location

    def set_ncol(self, nb_of_cols):
        self.ncol = nb_of_cols

    def set_box_anchor(self, anchor):
        self.box_anchor = anchor

    def set_fontsize(self, fontsize):
        self.fontsize = fontsize

    def set_fancybox(self, fancybox):
        self.fancybox = fancybox

    def set_shadow(self, shadow):
        self.shadow = shadow


class plot_settings(object):
    def __init__(self):
        self.title = None
        self.title_fontsize = 14

        self.legend = legend()

        self.xlabel = None
        self.ylabel = None
        self.xlabel_fontsize = 12
        self.ylabel_fontsize = 12

        self.xticks_fontsize = 11
        self.yticks_fontsize = 11

        # auto-set y-limits
        self.ylim_min = None
        self.ylim_max = None

        # auto-set x-limits
        self.xlim_min = None
        self.xlim_max = None

    def set_title(self, title):
        self.title = title

    def set_title_fontsize(self, fontsize):
        self.title_fontsize = fontsize

    def set_legend(self, legend):
        self.legend = legend

    def set_xlabel(self, xlabel):
        self.xlabel = xlabel

    def set_ylabel(self, ylabel):
        self.ylabel = ylabel

    def set_xlabel_fontsize(self, fontsize):
        self.xlabel_fontsize = fontsize

    def set_ylabel_fontsize(self, fontsize):
        self.ylabel_fontsize = fontsize

    def set_xticks_fontsize(self, fontsize):
        self.xticks_fontsize = fontsize

    def set_yticks_fontsize(self, fontsize):
        self.yticks_fontsize = fontsize

    def set_ylim(self, min, max):
        self.ylim_min = min
        self.ylim_max = max

    def set_xlim(self, min, max):
        self.xlim_min = min
        self.xlim_max = max


def x_iter_schedule(duration):
    x_iter = 10

    if duration < 10:
        x_iter = 1
    elif duration < 20:
        x_iter = 2
    elif duration < 50:
        x_iter = 5
    elif duration < 100:
        x_iter = 10
    elif duration < 1000:
        x_iter = 50
    else:
        x_iter = 100

    return x_iter


def boxplot(data_to_plot, title="Boxplot", xticklabels=None,
            horizontal_box=False, show=False, save=False):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)

    if horizontal_box:
        ## add patch_artist=True option to ax.boxplot()
        ## to get fill color
        bp = ax.boxplot(data_to_plot, patch_artist=True, vert=0)
    else:
        ## add patch_artist=True option to ax.boxplot()
        ## to get fill color
        bp = ax.boxplot(data_to_plot, patch_artist=True)

    # set tick labels
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    ### Styling
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=line_width)
        # change fill color
        box.set(facecolor='#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=line_width)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=line_width)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=line_width)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    if save:
        plt.savefig(
            os.path.join(PATH.FIGURES, '{}.pdf'.format(title)),
            bbox_inches='tight'
        )

    if show:
        plt.show()
        plt.close()

def plot_image(image, title="Image", save=False):
    img_rows, img_cols, nb_channels = image.shape

    if (nb_channels == 1):
        plt.imshow(image.reshape(img_rows, img_cols), cmap='gray')
    else:
        plt.imshow(image.reshape(img_rows, img_cols, nb_channels))

    plt.title(title)

    if save:
        plt.savefig(
            os.path.join(PATH.FIGURES, '{}.pdf'.format(title)),
            bbox_inches='tight'
        )
    plt.show()
    plt.close()

def plot_difference(controls, treatments, title="None", save=False):
    """
    Plot the original image, corresponding perturbed image, and their difference.
    :param controls:
    :param treatments:
    :param title:
    :return:
    """
    img_rows, img_cols, nb_channels = controls.shape[1:4]
    print('shapes: control_set - {}; treatment_set - {}'.format(controls.shape, treatments.shape))
    print('rows/cols/channels: {}/{}/{}'.format(img_rows, img_cols, nb_channels))

    pos = 1
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.02, wspace=0.001, hspace=0.015)

    fig.suptitle(title)
    cols = 3
    rows = 5

    diffs = controls - treatments
    for i in range(0, 5):
        # original image
        ax_orig = fig.add_subplot(rows, cols, pos)
        ax_orig.axis('off')
        ax_orig.grid(b=None)
        ax_orig.set_aspect('equal')
        if (nb_channels == 1):
            plt.imshow(controls[i].reshape(img_rows, img_cols), cmap='gray')
        else:
            plt.imshow(controls[i].reshape(img_rows, img_cols, nb_channels))
        pos += 1

        # transformed/perturbed image
        ax_changed = fig.add_subplot(rows, cols, pos)
        ax_changed.axis('off')
        ax_changed.grid(b=None)
        ax_changed.set_aspect('equal')
        if (nb_channels == 1):

            plt.imshow(treatments[i].reshape(img_rows, img_cols), cmap='gray')
        else:
            plt.imshow(treatments[i].reshape(img_rows, img_cols, nb_channels))
        pos += 1
        # difference
        ax_diff = fig.add_subplot(rows, cols, pos)
        ax_diff.axis('off')
        ax_diff.grid(b=None)
        ax_diff.set_aspect('equal')
        if (nb_channels == 1):
            plt.imshow(diffs[i].reshape(img_rows, img_cols), cmap='gray')
        else:
            plt.imshow(diffs[i].reshape(img_rows, img_cols, nb_channels))
        pos += 1

    if save:
        fig.savefig(
            os.path.join(PATH.FIGURES, '{}.pdf'.format(title)),
            bbox_inches='tight'
        )

    plt.show()
    plt.close()

def plot_comparisons(controls, treatments, title="None", save=False):
    """
    Draw some comparisons of original images and transformed/perturbed images.
    :param controls: the original images
    :param treatments: the transformed or perturbed images
    :return: na
    """
    img_rows, img_cols, nb_channels = controls.shape[1:4]
    print('shapes: control_set - {}; treatment_set - {}'.format(controls.shape, treatments.shape))
    print('rows/cols/channels: {}/{}/{}'.format(img_rows, img_cols, nb_channels))

    pos = 1
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.02, wspace=0.001, hspace=0.015)

    fig.suptitle(title)
    cols = 4
    rows = 5

    for i in range(1, 11):
        ax1 = fig.add_subplot(rows, cols, pos)
        ax1.axis('off')
        ax1.grid(b=None)
        ax1.set_aspect('equal')
        # show an original image
        if (nb_channels == 1):
            plt.imshow(controls[i - 1].reshape(img_rows, img_cols), cmap='gray')
        else:
            plt.imshow(controls[i - 1].reshape(img_rows, img_cols, nb_channels))
        pos += 1
        ax2 = fig.add_subplot(rows, cols, pos)
        ax2.axis('off')
        ax2.grid(b=None)
        ax2.set_aspect('equal')
        # show a transformed/perturbed images
        if (nb_channels == 1):
            plt.imshow(treatments[i - 1].reshape(img_rows, img_cols), cmap='gray')
        else:
            plt.imshow(treatments[i - 1].reshape(img_rows, img_cols, nb_channels))
        pos += 1

    if save:
        fig.savefig(
            os.path.join(PATH.FIGURES, '{}.pdf'.format(title)),
            bbox_inches='tight'
        )
    plt.show()
    plt.close()

def plot_lines(data, first_key_as_xlabel=True, setting=plot_settings(), save=False):
    """
    Plot curves in one figure, values[keys[i]] vs. values[keys[0]], where i > 0.
    Usage:
    Check eval_acc_upperbound() function in ../scripts/eval_model.py as an example.
    :param data: a dictionary. Data of the curves to plot, where
            (1) values of keys[0] is the value of x-axis
            (2) values of the rest keys are values of y-axis of each curve.
    :param title: string. Title of the figure.
    :param ylabel: string. The label of y-axis.
    :param save: boolean. Save the figure or not.
    :param legend_loc: location of legend
    :return:
    """
    nb_dimensions = len(data.keys())
    keys = list(data.keys())

    for i in range(1, nb_dimensions):
        m_id = (i - 1) % nb_marks
        c_id = (i - 1) % nb_colors
        m = '{}{}'.format(line_styles[0], marks[m_id])
        plt.plot(data[keys[0]], data[keys[i]], m, color=colors[c_id],
                 label=keys[i], linewidth=line_width)

    plt.title(setting.title)

    if first_key_as_xlabel:
        plt.xlabel(keys[0])
    else:
        plt.xlabel(setting.xlabel)

    plt.ylabel(setting.ylabel)
    plt.legend(loc=setting.legend.location)

    if save:
        plt.savefig(
            os.path.join(PATH.FIGURES, '{}.pdf'.format(setting.title)),
            bbox_inches='tight'
        )

    plt.show()
    plt.close()

def plot_scatter_with_certainty(data, filling_borders, setting=plot_settings(),
                                first_key_as_xlabel=True, save=False):
    """
    Plot lines and some filled areas.
    :param data: dictionary. data used to plot lines.
    :param filling_borders: list of tuples. each tuple consists of a sequence of values and corresponding certainties.
    :param title:
    :param ylabel:
    :param save:
    :param legend_loc: location of legend.
    :return:
    """
    nb_dimensions = len(data.keys())
    keys = list(data.keys())

    print('keys: ', keys)
    print('keys[0]: ', data[keys[0]])

    # plot lines
    for i in range(1, nb_dimensions):
        m_id = (i - 1) % nb_marks
        c_id = (i - 1) % nb_colors
        # mark = '{}{}'.format(line_styles[0], marks[m_id])
        mark = line_styles[(i - 1) % len(line_styles)]
        alpha = 0.9
        # if (headers.GAP.value == keys[i]):
        #     mark = line_styles[-1]
        #     alpha = 0.6

        plt.plot(data[keys[i]], mark, markerfacecolor='white',
                 color=colors[c_id], label=keys[i], alpha=alpha,
                 markersize=(line_width + 1), linewidth=line_width)

    # fill areas
    if filling_borders:
        nb_filling_areas = len(filling_borders)
        for i in range(nb_filling_areas):
            x, upper_bound, lower_bound = filling_borders[i]
            x1 = [a - 1 for a in x]
            # x1 = x
            print('upper_bound:', upper_bound)
            print('lower_bound:', lower_bound)

            plt.fill_between(x1, lower_bound, upper_bound, color='silver', alpha=.50)

    if setting.title is not None:
        plt.title(setting.title, fontsize=setting.title_fontsize)

    xticks = []
    xticks_labels = []

    iter = x_iter_schedule(setting.xlim_max - setting.xlim_min)

    for i in data[keys[0]]:
        i = int(i)

        if 0 == i % iter:
            xticks.append(i-1)
            xticks_labels.append(i)

    plt.xticks(xticks, xticks_labels, fontsize=setting.xticks_fontsize)
    plt.yticks(fontsize=setting.yticks_fontsize)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if first_key_as_xlabel:
        # get xlabel from data
        plt.xlabel(keys[0].replace('_', ' '), fontsize=setting.xlabel_fontsize)
    elif setting.xlabel is not None:
        plt.xlabel(setting.xlabel, fontsize=setting.xlabel_fontsize)

    if setting.ylabel is not None:
        plt.ylabel(setting.ylabel, fontsize=setting.ylabel_fontsize, )

    if setting.ylim_min is not None and setting.ylim_max is not None:
        plt.ylim(setting.ylim_min, setting.ylim_max)

    if setting.xlim_min is None or setting.xlim_max is None:
        plt.xlim(0.0, len(data[keys[0]]) - 1)
    else: # auto set x-limits
        plt.xlim(setting.xlim_min, setting.xlim_max)

    plt.legend(loc=setting.legend.location, bbox_to_anchor=setting.legend.box_anchor,
               fontsize=setting.legend.fontsize, ncol=setting.legend.ncol,
               fancybox=setting.legend.fancybox, shadow=setting.legend.shadow)

    plt.subplots_adjust(left=0.18, right=0.90, top=0.90, bottom=0.15)

    if save:
        plt.savefig(
            os.path.join(PATH.FIGURES, '{}.pdf'.format(setting.title)),
            bbox_inches='tight'
        )

    plt.show()
    plt.close()


def plot_training_history(history, model_name):
    fig = plt.figure(figsize=(1, 2))
    plt.subplots_adjust(left=0.05, right=0.95,
                        top=0.90, bottom=0.05,
                        wspace=0.01, hspace=0.01)

    # plot accuracies
    fig_acc = fig.add_subplot(111)
    fig_acc.plot(history['acc'])
    fig_acc.plot(history['val_acc'])
    fig_acc.plot(history['adv_acc'])
    fig_acc.plot(history['adv_val_acc'])
    fig_acc.title('Accuracy History')
    fig_acc.ylabel('Accuracy')
    fig_acc.xlabel('Epoch')
    fig_acc.legend(['train (legitimates), test (legitimates), train (adversarial), test (adversarial)'],
                   loc='upper left')

    # plot loss
    fig_loss = fig.add_subplot(122)
    fig_loss.plot(history['loss'])
    fig_loss.plot(history['val_loss'])
    fig_loss.plot(history['adv_loss'])
    fig_loss.plot(history['adv_val_loss'])
    fig_loss.title('Loss History')
    fig_loss.ylabel('Loss')
    fig_loss.xlabel('Epoch')
    fig_loss.legend(['train (legitimates), test (legitimates), train (adversarial), test (adversarial)'],
                   loc='upper left')

    # save the figure to a pdf
    fig.savefig(os.path.join(PATH.FIGURES, 'hist_{}.pdf'.format(model_name)), bbox_inches='tight')


def plotTrainingResult(history, model_name):
    # Plot training & validation accuracy values
    print("plotting accuracy")
    f_acc = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if MODE.DEBUG:
        plt.show()
    f_acc.savefig(
            os.path.join(PATH.FIGURES, model_name+"_training_acc_vs_val_acc.pdf"),
            bbox_inches='tight')
    plt.close()
    
    # Plot training & validation loss values
    print("plotting loss")
    f_loss = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if (MODE.DEBUG):
        plt.show()
    f_loss.savefig(
            os.path.join(PATH.FIGURES, model_name+"_training_loss_vs_val_loss.pdf"),
            bbox_inches='tight')
    plt.close()
