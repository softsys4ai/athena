"""
Implement methods plotting and drawing figures.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com))
        Jianhai Su
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import PATH, MODE

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


def plot_curves(data, title='curves', ylabel='None', save=False):
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
    :return:
    """
    nb_dimensions = len(data.keys())
    keys = list(data.keys())

    for i in range(1, nb_dimensions):
        plt.plot(data[keys[0]], data[keys[i]], '-+', label=keys[i])

    plt.xlabel(keys[0])
    plt.ylabel(ylabel)
    plt.legend(loc="lower right")

    if save:
        plt.savefig(
            os.path.join(PATH.FIGURES, '{}.pdf'.format(title)),
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
