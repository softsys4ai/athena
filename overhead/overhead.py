import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({'font.size': 16})
mpl.rc('lines', markersize=14)
mpl.rc('boxplot.flierprops', markersize=14)


result_dir = sys.argv[1]

# transformation time in each of 72 weak defenses
# shape of transTC: (72).
# 72 weak defense
transTC=np.load("transTCs.npy")
# inference time in each of 72 weak defenses
# shape of predTC: (72)
predTC=np.load("predTCs.npy")
# time cost of ensemble calculation
# shape of ensembleTC: (5)
# five ensembles
ensembleTC=np.load("ensembleTCs.npy")
# inference time of the original model
predTCOnOriModel = np.load("predTCOnOriModel.npy")




AE_Type = '{} ({})'.format(r'$FGSM$', r'$\epsilon: 0.3$')



ind=np.argsort(TransTC, axis=1)

# Overhead: ensemble cost vs inference cost
fig1, ax1 = plt.subplots()
ax1.set_ylabel(r"Averaged Time Cost ($log(t)$ in $\mu s$)")
bp1 = ax1.boxplot((6+np.log10(predTC)).tolist(), notch=False, patch_artist=True,
            boxprops=dict(facecolor="C1"), widths=0.20,
           medianprops=dict(color='C0'))
oriM_plot, = ax1.plot(list(range(1, 2)), 6+np.log10(predTCOnOriModel), 'bv', label="Original Model")
rd_plot, = ax1.plot(list(range(1, 2)), 6+np.log10(ensembleTC[0]), 'k*', label="ENS(RD)")
mv_plot, = ax1.plot(list(range(1, 2)), 6+np.log10(ensembleTC[1]), 'o', markeredgewidth=1, markeredgecolor='g',
markerfacecolor='None', label="ENS(MV)")
avep_plot, = ax1.plot(list(range(1, 2)), 6+np.log10(ensembleTC[2]), 's', markeredgewidth=1,markeredgecolor='y',
markerfacecolor='None', label="ENS(AVEP)")
t2mv_plot, = ax1.plot(list(range(1, 2)), 6+np.log10(ensembleTC[3]), 'rx', label="ENS(T2MV)")
avel_plot, = ax1.plot(list(range(1, 2)), 6+np.log10(ensembleTC[4]), '+', markeredgewidth=1,markeredgecolor='c',
markerfacecolor='None', label="ENS(AVEL)")
plt.xticks([], [])
#plt.xticks(list(range(1, 11)), titles_for_attacks2, rotation=90)
leg1 = ax1.legend(
    [rd_plot, mv_plot, avep_plot, t2mv_plot, avel_plot],
    ["ENS(RD)", "ENS(MV)", "ENS(AVEP)", "ENS(T2MV)", "ENS(AVEL)"],
    loc='best', bbox_to_anchor=(0.92, 0.32, 1, 0.28), title="Ensemble Strategy")

ax1.legend([bp1["boxes"][0], oriM_plot], ['Transformation\nModels', 'Original Model'], loc='best',
           bbox_to_anchor=(0.92, 0.73, 1, 0.28), title="Inference Time")
ax1.add_artist(leg1)
plt.subplots_adjust(left=0.45, right=0.55, top=1.0, bottom=0)
#plt.tight_layout()
fig1.savefig(
        os.path.join(result_dir, "time_cost_ENS_vs_Inference_FGSM300.pdf"),
        dpi=1200, bbox_inches='tight')


# Transformation vs Inference
plt.rcParams.update({'font.size': 14})
ind=np.argsort(sTransTC, axis=1)
ae_idx=3 # FGSM eps=0.3

plt.plot(range(1, 73), np.log10(1000000*(transTC[ind])), label="Transformation")
plt.plot(range(1, 73), np.ones(72)*np.log10(1000000*predTCOnOriModel), 'r', label = "Inference of the Original Model")
plt.xlabel("ID of Sorted Weak Defenses")
plt.ylabel(r"Averaged Time Cost ($log(t)$ in $\mu s$)")
plt.legend()
plt.savefig(
        os.path.join(result_dir, "overhead_FGSM300.pdf"),
        dpi=1200, bbox_inches='tight')
#plt.show()



