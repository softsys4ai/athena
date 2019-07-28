"""
Implement methods plotting and drawing figures.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com))
"""
import matplotlib.pyplot as plt
from config import DATA

def draw_comparisons(controls, treatments):
    """
    Draw some comparisons of original images and transformed/perturbed images.
    :param controls: the original images
    :param treatments: the transformed or perturbed images
    :return: na
    """
    pos = 1
    fig = plt.figure(figsize=(10, 10))
    cols = 4
    rows = 5

    for i in range(1, 11):
        fig.add_subplot(rows, cols, pos)
        # show an original image
        plt.imshow(controls[i - 1].reshape(DATA.IMG_ROW, DATA.IMG_COL), cmap='gray')
        pos += 1
        fig.add_subplot(fig.add_subplot(rows, cols, pos))
        # show a transformed/perturbed images
        plt.imshow(treatments[i - 1].reshape(DATA.IMG_ROW, DATA.IMG_COL), cmap='gray')
        pos += 1
    plt.show()
