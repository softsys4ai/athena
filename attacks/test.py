import attacker, whitebox

from data import load_data
import numpy as np

model_name = 'mnist'
attack_method = 'MIM'
X, Y = load_data('mnist')

X_adv, Y = attacker.get_adversarial_examples(model_name=model_name, attack_method=attack_method, X=X, Y=Y)

for index, adv in enumerate(X_adv):
    np.save(f'MIM_example_{index}', adv)
