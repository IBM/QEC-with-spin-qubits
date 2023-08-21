# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import sys
import plot_utils
import numpy as np
import json
import time
from multiprocessing import Pool

from RotatedSurfaceCode import RotatedSurfaceCode


Nphi = 60

#pG = 0.001
#pTmax, pRmax = 0.04,0.07
#noisyLogRO = True
#Num_p,delpth = 100, 0.5

pG = 0
pTmax, pRmax = 0.06,0.17
noisyLogRO = False
Num_p,delpth = 70, 0.6

Gbias, Tbias = 1, 20
dmin,dmax = 11,17
logical = 'X'
max_shots, shot_batch, max_num_fail, max_fail_rate = 300_000, 3_000, 30_000, 0.4


slist = np.linspace(0,1,Nphi)
pG_list,pT_list,pR_list = np.transpose([([pG,s*pTmax,(1-s)*pRmax]) for s in slist])
phi_list = np.arctan2(pT_list,pG_list)
theta_list = np.arctan2(pR_list,np.sqrt(pT_list**2 + pG_list**2))

results_threshold = []
results_full = []
outpath_threshold = '/u/bhetenyi/results/RSC_pTpR_threshold_pG0_logical'+logical+'_dmax'+str(dmax)+'_maxshots'+str(int(max_shots/1000))+'k_Gbias'+str(Gbias)+'_Tbias'+str(Tbias)+'_noisyLogRO_'+str(noisyLogRO)+'.json'
outpath_full = '/u/bhetenyi/results/RSC_pTpR_full_threshold_pG0_plots_logical'+logical+'_dmax'+str(dmax)+'_maxshots'+str(int(max_shots/1000))+'k_Gbias'+str(Gbias)+'_Tbias'+str(Tbias)+'_noisyLogRO_'+str(noisyLogRO)+'.json'
p_threshold_guess = np.sqrt(pG_list**2 + pT_list**2 + pR_list**2)

def LogFail_of_d_p_arglist(arglist):
    return plot_utils.LogFail_of_d_p(RotatedSurfaceCode,
                                    theta = arglist[0], phi = arglist[1], Gbias = Gbias, Tbias = Tbias, logical = logical,
                                    dist_list = range(dmin,dmax+2,2), error_list = arglist[2]*np.linspace(1.-delpth,1.+delpth,Num_p), T_over_d = 1,
                                    max_shots = max_shots, shot_batch = shot_batch, max_num_fail = max_num_fail, max_fail_rate = max_fail_rate, noisyLogRO = noisyLogRO)

if __name__ == '__main__':
    with Pool(Nphi) as p:
            t1 = time.time()
            results_full_threshold_plots_of_theta_phi = p.map(LogFail_of_d_p_arglist,
                                                                    [arglist for arglist in zip(
                                                                    theta_list,
                                                                    phi_list,
                                                                    p_threshold_guess)]
                                                                    )
            t2 = time.time()
            print("time: ",t2-t1)
            result_thresholds_in_row = []
            results_full_of_phi_for_theta = []

            for theta,phi,Log_fail_d_p in results_full_threshold_plots_of_theta_phi:
                p_threshold, p_threshold_error = plot_utils.Threshold_from_LogFail(Log_fail_d_p)
                result_thresholds_in_row.append([theta,phi,p_threshold,p_threshold_error])
                print("theta, phi = ",np.round(theta,3),np.round(phi,3)," p_th = ",np.round(p_threshold,3),"+/-",np.round(p_threshold_error,5))

            results_threshold.append(result_thresholds_in_row)
            results_full.append(results_full_threshold_plots_of_theta_phi)

            with open(outpath_threshold, "w") as f:
                f.write(json.dumps(results_threshold))
            with open(outpath_full, "w") as f:
                f.write(json.dumps(results_full))
