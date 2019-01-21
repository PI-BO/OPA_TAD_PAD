import pickle
import matplotlib.pyplot as plt
import numpy as np


fontsize = 18
## peak time
with open('result/exp_speed.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, \
    losses_linear, losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)


plt.plot(anonymity_levels, losses_best,'o-', label='Ground truth metric')
plt.plot(anonymity_levels, losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_levels, np.mean(losses_linear, axis=1), np.std(losses_linear, axis=1),
             fmt='^--', capthick=2,label='Linear metric')
plt.errorbar(anonymity_levels, np.mean(losses_deep, axis=1), np.std(losses_deep, axis=1),
             fmt='X--',capthick=2,label='Nonlinear metric')
plt.xlabel('Anonymity level', fontsize=18)
plt.ylabel('Information loss', fontsize=18)
plt.title('Speed usage example', fontsize=18)
plt.legend()
plt.show()


# peak time (presanitized)
with open('result/exp_speed_presanitized.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, \
    losses_linear, losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)

plt.plot(anonymity_levels, losses_best,'o-', label='Ground truth metric')
plt.plot(anonymity_levels, losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_levels, np.mean(losses_linear, axis=1), np.std(losses_linear, axis=1),
             fmt='^--', capthick=2,label='Linear metric')
plt.errorbar(anonymity_levels, np.mean(losses_deep, axis=1), np.std(losses_deep, axis=1),
             fmt='X--',capthick=2,label='Nonlinear metric')
plt.xlabel('Anonymity level', fontsize=18)
plt.ylabel('Information loss', fontsize=18)
plt.title('Speed usage example', fontsize=18)
plt.legend()
plt.show()
