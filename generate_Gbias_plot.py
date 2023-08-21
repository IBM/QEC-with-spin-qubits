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
Tbias = float(kwarg_dict['Tbias'])
Num_p = int(kwarg_dict['Num_p'])
p_threshold_guess = float(kwarg_dict['pthguess']) 
delpth = float(kwarg_dict['delpth']) #relative deviation from linearized threshold
max_shots = int(kwarg_dict['max_shots'])
shot_batch = int(kwarg_dict['shot_batch'])
max_num_fail = int(kwarg_dict['max_num_fail']) 
max_fail_rate = float(kwarg_dict['max_fail_rate'])

results_threshold = []
results_full = []
outpath_threshold = '~/'+kwarg_dict['code']+'_threshold_Gbias_logical'+logical+'_dmax'+kwarg_dict['dmax']+'_maxshots'+str(int(max_shots/1000))+'k_Tbias'+kwarg_dict['Tbias']+'.json'
outpath_full = '~/'+kwarg_dict['code']+'_full_threshold_plots_Gbias_logical'+logical+'_dmax'+kwarg_dict['dmax']+'_maxshots'+str(int(max_shots/1000))+'k_Tbias'+kwarg_dict['Tbias']+'.json'

Gbias_list = np.logspace(-2,2,21)
Nphi = len(Gbias_list)

def LogFail_of_d_p_Gbias(Gbias):
    return plot_utils.LogFail_of_d_p(code,
                                    theta = 0, phi = 0, Gbias = Gbias, Tbias = Tbias, logical = logical,
                                    dist_list = range(dmin,dmax+2,2), error_list = p_threshold_guess*np.linspace(1.-delpth,1.+delpth,Num_p), T_over_d = 1,
                                    max_shots = max_shots, shot_batch = shot_batch, max_num_fail = max_num_fail, max_fail_rate = max_fail_rate)

if __name__ == '__main__':
    with Pool(Nphi) as p:
            t1 = time.time()
            results_full_threshold_plots_of_theta_phi = p.map(LogFail_of_d_p_Gbias, [Gbias for Gbias in Gbias_list])
            t2 = time.time()
            print("time: ",t2-t1)
            result_thresholds_in_row = []
            results_full_of_Gbias = []

            for Gbias,[theta,phi,Log_fail_d_p] in zip(Gbias_list, results_full_threshold_plots_of_theta_phi):
                p_threshold, p_threshold_error = plot_utils.Threshold_from_LogFail(Log_fail_d_p)
                result_thresholds_in_row.append([Gbias,p_threshold,p_threshold_error])
                results_full_of_Gbias.append([Gbias,Log_fail_d_p])
                print("Gbias = ",np.round(Gbias,3)," p_th = ",np.round(p_threshold,3),"+/-",np.round(p_threshold_error,5))

            results_threshold.append(result_thresholds_in_row)
            results_full.append(results_full_of_Gbias)

            with open(outpath_threshold, "w") as f:
                f.write(json.dumps(results_threshold))
            with open(outpath_full, "w") as f:
                f.write(json.dumps(results_full))
