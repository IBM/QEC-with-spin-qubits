
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
import os


Nphi = 20
Nlim = int(Nphi*(Nphi+1)/2)

data_to_merge = [['RSC','17','300k',['X','Z']], ['XZZX','17','300k',['X','Z']], ['SC3','17','300k',['X','Z']], ['XYZ2','17','300k',['Y','Z']], ['FCC','8','300k',['X','Z']], ['HFC','8','300k',['X','Z']]]

for code, dmax, maxshots, logicals in data_to_merge:
    for logical in logicals:
        merged_list =[]
        merged_list2 =[]
        home = '~/'
        fname = code+'_threshold_3d_logical'+logical+'_dmax'+dmax+'_maxshots'+maxshots+'_Gbias1_Tbias20'
        fname_full = code+'_full_threshold_plots_3d_logical'+logical+'_dmax'+dmax+'_maxshots'+maxshots+'_Gbias1_Tbias20'

        for Nstart in range(0,Nlim,Nphi):
            file_name = home+fname+'_Nstart'+str(Nstart)+'_Nphi'+str(Nphi)+'.json'
            file_name2 = home+fname_full+'_Nstart'+str(Nstart)+'_Nphi'+str(Nphi)+'.json'
            merged_list.extend(json.load(open(file_name)))
            merged_list2.extend(json.load(open(file_name2)))

        with open(home+fname+'_all.json', "w") as f:
            f.write(json.dumps(merged_list))
        with open(home+fname_full+'_all.json', "w") as f:
            f.write(json.dumps(merged_list2))

        for Nstart in range(0,Nlim,Nphi):
            file_name = home+fname+'_Nstart'+str(Nstart)+'_Nphi'+str(Nphi)+'.json'
            file_name2 = home+fname_full+'_Nstart'+str(Nstart)+'_Nphi'+str(Nphi)+'.json'
            os.remove(file_name)
            os.remove(file_name2)
