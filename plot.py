"""
Implement methods plotting and drawing figures.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com))
"""
import matplotlib.pyplot as plt
from config import PATH

def draw_comparisons(controls, treatments, title="None"):
    """
    Draw some comparisons of original images and transformed/perturbed images.
    :param controls: the original images
    :param treatments: the transformed or perturbed images
    :return: na
    """
    img_rows, img_cols, nb_channels = controls.shape[1:4]
    pos = 1
    fig = plt.figure(figsize=(10, 10))


    fig.suptitle(title)
    cols = 4
    rows = 5

    for i in range(1, 11):
        ax1 = fig.add_subplot(rows, cols, pos)
        ax1.axis('off')
        ax1.grid(b=None)
        ax1.set_aspect('equal')
        # show an original image
        plt.imshow(controls[i - 1].reshape(img_rows, img_cols), cmap='gray')
        pos += 1
        ax2 = fig.add_subplot(fig.add_subplot(rows, cols, pos))
        ax2.axis('off')
        ax2.grid(b=None)
        ax2.set_aspect('equal')
        # show a transformed/perturbed images
        plt.imshow(treatments[i - 1].reshape(img_rows, img_cols), cmap='gray')
        pos += 1

    plt.subplots_adjust(wspace=0.01, hspace=0.05)
    plt.show()
    fig.savefig('{}/{}.jpg'.format(PATH.FIGURES, title))