import sys
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from rank_correlation.heatmap import corrplot

np.random.seed(127)
sns.set()

# file path: each weak defense' accuracy over 27 types of AEs
acc_FP=sys.argv[1]
resultDir=sys.argv[2]

# Rank Correlation
df=pd.read_csv(acc_FP, encoding='cp1252')
acc_df = df.copy()
acc_df = acc_df.drop(["ID", "Model", "BS"], axis=1)

print("Apply Spearman Rank Correlation and Hierarchical Clustering")
rank_corr = acc_df.corr(method='spearman')
rank_corr.to_csv(os.path.join(resultDir, "./rank_correlation.csv"))

titles_for_attacks = {
     "biml2(eps:0.75)":'{} ({})'.format(r'$BIM\_l_2$', r'$\epsilon: 0.75$'),
     "biml2(eps:1.0)":'{} ({})'.format(r'$BIM\_l_2$', r'$\epsilon: 1.0$'),
     "biml2(eps:1.2)":'{} ({})'.format(r'$BIM\_l_2$', r'$\epsilon: 1.2$'),
     "bimli(eps:0.075)":'{} ({})'.format(r'$BIM\_l_{\infty}$', r'$\epsilon: 0.075$'),
     "bimli(eps:0.09)":'{} ({})'.format(r'$BIM\_l_{\infty}$', r'$\epsilon: 0.09$'),
     "bimli(eps:0.12)":'{} ({})'.format(r'$BIM\_l_{\infty}$', r'$\epsilon: 0.12$'),
     "cwl2(lr:0.01)":'{} ({})'.format(r'$CW\_l_2$', r'$lr: 0.01$'),
     "cwl2(lr:0.012)":'{} ({})'.format(r'$CW\_l_2$', r'$lr: 0.012$'),
     "cwl2(lr:0.015)":'{} ({})'.format(r'$CW\_l_2$', r'$lr: 0.015$'),
     "dfl2(os:3/255)":'{} ({})'.format(r'$DF\_l_2$', r'$overshoot: 3$'),
     "dfl2(os:8/255)":'{} ({})'.format(r'$DF\_l_2$', r'$overshoot: 8$'),
     "dfl2(os:20/255)":'{} ({})'.format(r'$DF\_l_2$', r'$overshoot: 20$'),
     "fgsm(eps:0.1)":'{} ({})'.format(r'$FGSM$', r'$\epsilon: 0.1$'),
     "fgsm(eps:0.2)":'{} ({})'.format(r'$FGSM$', r'$\epsilon: 0.2$'),
     "fgsm(eps:0.3)":'{} ({})'.format(r'$FGSM$', r'$\epsilon: 0.3$'),
     "jsma(theta:0.15)":'{} ({})'.format(r'$JSMA$', r'$\theta: 0.15$'),
     "jsma(theta:0.18)":'{} ({})'.format(r'$JSMA$', r'$\theta: 0.18$'),
     "jsma(theta:0.21)":'{} ({})'.format(r'$JSMA$', r'$\theta: 0.21$'),
     "mim(eps:0.05)":'{} ({})'.format(r'$MIM$', r'$\epsilon: 0.05$'),
     "mim(eps:0.075)":'{} ({})'.format(r'$MIM$', r'$\epsilon: 0.075$'),
     "mim(eps:0.1)":'{} ({})'.format(r'$MIM$', r'$\epsilon: 0.1$'),
     "onepixel(pxCnt:5)":'{} ({})'.format(r'$OP$', r'$px~count: 5$'),
     "onepixel(pxCnt:15)":'{} ({})'.format(r'$OP$', r'$px~count: 15$'),
     "onepixel(pxCnt:30)":'{} ({})'.format(r'$OP$', r'$px~count: 30$'),
     "pgd(eps:0.075)":'{} ({})'.format(r'$PGD$', r'$\epsilon: 0.075$'),
     "pgd(eps:0.09)":'{} ({})'.format(r'$PGD$', r'$\epsilon: 0.09$'),
     "pgd(eps:0.1)":'{} ({})'.format(r'$PGD$', r'$\epsilon: 0.1$'),
     "BS":'Benign Samples',
}

rank_corr=rank_corr.rename(columns=titles_for_attacks)
rank_corr=rank_corr.rename(index=titles_for_attacks)

filepath=os.path.join(resultDir, "rank_correlation_plot.pdf")
plt.figure(figsize=(50, 50))
corrplot(rank_corr, filepath, size_scale=36, palette=sns.diverging_palette(5, 250, n=256))


