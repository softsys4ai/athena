from attacks import attacker

from config import *
import data

if __name__ == '__main__':
    (X, Y) = data.load_data(DATA.mnist)[1]
    X_adv, Y = attacker.get_adversarial_examples('../data/models/model-mnist-cnn-clean.h5',
                                                 attack_method=ATTACK.ONE_PIXEL,
                                                 X=X,
                                                 Y=Y)

