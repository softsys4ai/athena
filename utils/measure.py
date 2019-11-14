"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from numpy import linalg as LA

def frobenius_norm(X1, X2):
    """
    Compute the average L_ord Norm between 2 data sets.
    :param X1:
    :param X2:
    :param ord: order of the norm. l0, l2, linf
    :return:
    """
    ord = 2
    if len(X1.shape) < 4:
        nb_samples = 1
        img_rows = X1[0]
        img_cols = X1.shape[1]
    else:
        nb_samples = X1.shape[0]
        img_rows = X1.shape[1]
        img_cols = X1.shape[2]

    norm = 0.
    for x1, x2 in zip(X1, X2):

        x1 = x1.reshape(img_rows, img_cols)
        x2 = x2.reshape(img_rows, img_cols)
        perturbation = abs(x1 - x2)
        norm += LA.norm(perturbation, ord=ord) / LA.norm(x2, ord=ord)

    norm = float('{:.4f}'.format(norm/nb_samples))

    print('Distance(norm-{}): {}'.format(ord, norm))
    return norm