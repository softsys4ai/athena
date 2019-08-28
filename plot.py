"""
Implement methods plotting and drawing figures.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com))
        Jianhai Su
"""
import os
import matplotlib.pyplot as plt
from config import PATH, MODE

def draw_comparisons(controls, treatments, title="None"):
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
        if (treatments.shape[3] == 1) or ((nb_channels > 1) and (treatments.shape[3] == 1)):
            plt.imshow(treatments[i - 1].reshape(img_rows, img_cols), cmap='gray')
        else:
            plt.imshow(treatments[i - 1].reshape(img_rows, img_cols, treatments.shape[3]))
        pos += 1

    fig.savefig(
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
