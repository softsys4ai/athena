# Calculate and plot the dissimilarity between BS and each type of AEs

import sys
import os
import numpy as np
from numpy import linalg as LA


def distortion(bs_fp, ae_fp):
    bs = np.load(bs_fp)
    ae = np.load(ae_fp)

    bs = bs[:, :, :, 0]
    ae = ae[:, :, :, 0]

    diff = bs - ae
    ns = LA.norm(diff, axis=0)

    return round(ns.mean(), 2)

# directory that contains Bening samples and all types of AEs
sample_dir = sys.argv[1]
result_dir=sys.argv[2]
fs = os.listdir(sample_dir)
BS_fps = [x for x in fs if x[0:2]=="BS"]
BS_fp = BS_fps[0]
AE_fps = [x for x in fs if x[0:2]=="AE"]

fp = os.path.join(result_dir, "dissimilarity.csv")
with open(fp, "w") as fp:
    fp.write("AEType\tDissimilarity\n")
    for AE_fp in AE_fps:
        AE_type = AE_fp[19:-4]
        fp.write("{}\t{}\n".format(AE_type, distortion(BS_fp, AE_fp)))

