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
from SurfaceCode3CX import SurfaceCode3CX
from XZZXCode import XZZXCode
from HeavyHEXFloquetColorCode import HeavyHexFloquetColorCode
from HeavyHEXHoneycombFloquetCode import HeavyHexHoneycombFloquetCode
from XYZ2Code import XYZ2Code

kwarg_dict = dict(arg.split('=') for arg in sys.argv[1:])

code_dict = {'RSC': RotatedSurfaceCode, 'XZZX':XZZXCode, 'SC3': SurfaceCode3CX, 'XYZ2': XYZ2Code, 'HFC': HeavyHexHoneycombFloquetCode, 'FCC': HeavyHexFloquetColorCode}
code = code_dict[kwarg_dict['code']]
logical = kwarg_dict['logical']
dmin = int(kwarg_dict['dmin'])
dmax = int(kwarg_dict['dmax'])
Gbias = float(kwarg_dict['Gbias'])
Tbias = float(kwarg_dict['Tbias'])
pGmax = float(kwarg_dict['pGmax'])
pTmax = float(kwarg_dict['pTmax'])
pRmax = float(kwarg_dict['pRmax'])
Num_p = int(kwarg_dict['Num_p'])
delpth = float(kwarg_dict['delpth']) #relative deviation from linearized threshold
max_shots = int(kwarg_dict['max_shots'])
shot_batch = int(kwarg_dict['shot_batch'])
max_num_fail = int(kwarg_dict['max_num_fail']) 
max_fail_rate = float(kwarg_dict['max_fail_rate'])
Nphi = int(kwarg_dict['Nphi'])
Nstart = int(kwarg_dict['Nstart'])

slist = np.linspace(0,1,Nphi) 
tlist = np.linspace(0,1,Nphi)
pG_list,pT_list,pR_list = np.transpose([([(1-s-t)*pGmax,s*pTmax,t*pRmax]) for s in slist for t in tlist if s+t<=1])
phi_list = np.arctan2(pT_list,pG_list)
theta_list = np.arctan2(pR_list,np.sqrt(pT_list**2 + pG_list**2))

results_threshold = []
results_full = []
outpath_threshold = '~/'+kwarg_dict['code']+'_threshold_3d_logical'+logical+'_dmax'+kwarg_dict['dmax']+'_maxshots'+str(int(max_shots/1000))+'k_Gbias'+kwarg_dict['Gbias']+'_Tbias'+kwarg_dict['Tbias']+'_Nstart'+str(Nstart)+'_Nphi'+str(Nphi)+'.json'
outpath_full = '~/'+kwarg_dict['code']+'_full_threshold_plots_3d_logical'+logical+'_dmax'+kwarg_dict['dmax']+'_maxshots'+str(int(max_shots/1000))+'k_Gbias'+kwarg_dict['Gbias']+'_Tbias'+kwarg_dict['Tbias']+'_Nstart'+str(Nstart)+'_Nphi'+str(Nphi)+'.json'
p_threshold_guess = np.sqrt(pG_list**2 + pT_list**2 + pR_list**2)

def LogFail_of_d_p_arglist(arglist):
    return plot_utils.LogFail_of_d_p(code,
                                    theta = arglist[0], phi = arglist[1], Gbias = Gbias, Tbias = Tbias, logical = logical,
                                    dist_list = range(dmin,dmax+2,2), error_list = arglist[2]*np.linspace(1.-delpth,1.+delpth,Num_p), T_over_d = 1,
                                    max_shots = max_shots, shot_batch = shot_batch, max_num_fail = max_num_fail, max_fail_rate = max_fail_rate)

for batchind in range(Nstart,min(int(Nphi*(Nphi+1)/2),Nstart+2*Nphi), Nphi):
    if __name__ == '__main__':
        with Pool(Nphi) as p:
                t1 = time.time()
                results_full_threshold_plots_of_theta_phi = p.map(LogFail_of_d_p_arglist, 
                                                                        [arglist for arglist in zip(
                                                                        theta_list[batchind:batchind+Nphi],
                                                                        phi_list[batchind:batchind+Nphi],
                                                                        p_threshold_guess[batchind:batchind+Nphi])]
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
