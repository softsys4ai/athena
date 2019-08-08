"""
Implement methods plotting and drawing figures.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com))
"""
import matplotlib.pyplot as plt
from config import PATH

def draw_comparisons(title, controls, treatments):
    """
    Draw some comparisons of original images and transformed/perturbed images.
    :param controls: the original images
    :param treatments: the transformed or perturbed images
    :return: na
    """
    pos = 1
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)
    cols = 4
    rows = 5

    for i in range(1, 11):
        fig.add_subplot(rows, cols, pos)
        img_rows, img_cols = controls.shape[1:3]
        # show an original image
        plt.imshow(controls[i - 1].reshape(img_rows, img_cols), cmap='gray')
        pos += 1
        fig.add_subplot(fig.add_subplot(rows, cols, pos))
        # show a transformed/perturbed images
        plt.imshow(treatments[i - 1].reshape(img_rows, img_cols), cmap='gray')
        pos += 1
    plt.show()
    fig.savefig('{}/{}.jpg'.format(PATH.FIGURES, title))