# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from math import comb
from typing import *
import scipy.optimize as optimize
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def LogFail_of_d_p(code_circuit_class,
                   theta: float, phi: float, Gbias: float, Tbias: float, logical: str,
                   dist_list: List[int], error_list: List[float], T_over_d: float,
                   max_shots: int, shot_batch: int, max_num_fail: int, max_fail_rate: float = 0.49, **kwargs):
    '''For a given set of noise parameters: theta, phi, Gbias, Tbias the function calculates the log failure rates for ever distace in d_list, and error rate in error_list'''
    Log_fail_d_p = []
    for d in dist_list:
        Log_fail_p =[]
        for error in error_list:
            if len(Log_fail_p) == 0 or Log_fail_p[-1][1] < max_fail_rate:
                code = code_circuit_class(d=d, T=int(T_over_d*d), logical_observable=logical,
                                            gate_error_1q = error*2*np.cos(theta)*np.cos(phi)/(1+Gbias), gate_error_2q = error*2*np.cos(theta)*np.cos(phi)*Gbias/(1+Gbias),
                                            idle_error_T1 = error*np.cos(theta)*np.sin(phi)/(1+Tbias), idle_error_T2 = error*np.cos(theta)*np.sin(phi)*Tbias/(1+Tbias), 
                                            measurement_error_rate = error*np.sin(theta),**kwargs)
                stim_circuit = code.stim_circuit
                m = code.matching()
                
                num_correct = 0
                num_fail = 0
                num_shots = 0
                while num_shots<max_shots and num_fail<max_num_fail:
                    num_shots += shot_batch
                    detector_samples = stim_circuit.compile_detector_sampler().sample(shot_batch, append_observables=True) 
                    for sample in detector_samples:
                        actual_observable = sample[-1]
                        detectors_only = sample.copy()
                        detectors_only[-1] = 0
                        predicted_observable = code.PredictedObservableOutcome(sample=detectors_only,m=m)
                        num_fail += actual_observable != predicted_observable
                        num_correct += actual_observable == predicted_observable
                Log_fail_p.append([error,(num_shots-num_correct)/num_shots])
        Log_fail_d_p.append([d,Log_fail_p])

    return theta,phi,Log_fail_d_p

def Threshold_from_LogFail(Log_fail_d_p, error_list_cutoff: float = 0.495):
    # determine the largest error-list index of the shortest error list
    max_eind = len(Log_fail_d_p[0][1])
    for dind in range(len(Log_fail_d_p)):
        error_list, pLlist = np.transpose(Log_fail_d_p[dind][1])
        eind = 0
        while pLlist[eind]<error_list_cutoff and eind<len(error_list)-1:
            eind+=1
        if eind<max_eind:
            max_eind=eind

    error_list = [Log_fail_d_p[-1][1][i][0] for i in range(max_eind)] # redefine error_list as the shortest one
    dist_list = [Log_fail_d_p[dind][0] for dind in range(len(Log_fail_d_p))]
    
    rates_below_threshold = []
    rates_above_threshold = []
    pL_range = []
    for errorind in range(len(error_list)):
        pLofdist = [Log_fail_d_p[dind][1][errorind][1] for dind in range(len(dist_list))]
        pL_range.append(max(pLofdist) - min(pLofdist))
        if min(pLofdist) > 0.: # if the logical error rate is still 0 for some distance, we must be far from threshold
            if pLofdist[::-1] == sorted(pLofdist):
                rates_below_threshold.append(error_list[errorind]) # we only need the max
            if pLofdist == sorted(pLofdist):
                rates_above_threshold.append(error_list[errorind]) # we only need the min
    if rates_below_threshold == [] or rates_above_threshold == []:
        p_threshold,p_threshold_error = 0,0
    else:
        lower_bound_ind = error_list.index(max(rates_below_threshold))
        upper_bound_ind = error_list.index(min(rates_above_threshold))
        pL_range_lower = pL_range[lower_bound_ind]
        pL_range_upper = pL_range[upper_bound_ind]
        p_threshold = (pL_range_upper*max(rates_below_threshold)+pL_range_lower*min(rates_above_threshold))/(pL_range_lower + pL_range_upper)
        p_threshold_error = max(p_threshold-max(rates_below_threshold),min(rates_above_threshold)-p_threshold)

    return p_threshold, p_threshold_error

def Threshold_of_d(Log_fail_d_p):
    error_list = [Log_fail_d_p[-1][1][i][0] for i in range(len(Log_fail_d_p[-1][1]))] # redefine error_list as the last/shortest one
    dist_list = [Log_fail_d_p[dind][0] for dind in range(len(Log_fail_d_p))]
    
    rates_below_threshold = []
    rates_above_threshold = []
    p_threshold_of_d = []
    p_threshold_error_of_d = []
    for mindind in range(0,len(dist_list)-1):
        pL_range = []
        for errorind in range(len(error_list)):
            pLofdist = [Log_fail_d_p[dind][1][errorind][1] for dind in [mindind,mindind+1]]
            pL_range.append(max(pLofdist) - min(pLofdist))
            if min(pLofdist) > 0.: # if the logical error rate is still 0 for some distance, we must be far from threshold
                if pLofdist[::-1] == sorted(pLofdist):
                    rates_below_threshold.append(error_list[errorind]) # we only need the max
                if pLofdist == sorted(pLofdist):
                    rates_above_threshold.append(error_list[errorind]) # we only need the min
        if rates_below_threshold == [] or rates_above_threshold == []:
            p_threshold = 0
        else:
            lower_bound_ind = error_list.index(max(rates_below_threshold))
            upper_bound_ind = error_list.index(min(rates_above_threshold))
            pL_range_lower = pL_range[lower_bound_ind]
            pL_range_upper = pL_range[upper_bound_ind]
            p_threshold = (pL_range_upper*max(rates_below_threshold)+pL_range_lower*min(rates_above_threshold))/(pL_range_lower + pL_range_upper)
            p_threshold_error = max(p_threshold-max(rates_below_threshold),min(rates_above_threshold)-p_threshold)
        p_threshold_of_d.append(p_threshold)
        p_threshold_error_of_d.append(p_threshold_error)

    return p_threshold_of_d, p_threshold_error_of_d

def plot_3d_threshold(LogX_fail_3d_d_p,LogZ_fail_3d_d_p,ax,truncate=True,alpha=1,cmap='copper',interpolate=True,rescalepR = True):
    pG_list = []
    pT_list = []
    pR_list = []
    pth_list = []
    pG_listB = []
    pT_listB = []
    pR_listB = []
    pth_listB = []
    for rowind in range(len(LogX_fail_3d_d_p)):
        for columnind in range(len(LogX_fail_3d_d_p[rowind])):
            theta, phi, p_threshold1, _ = LogX_fail_3d_d_p[rowind][columnind]
            theta, phi, p_threshold2, _ = LogZ_fail_3d_d_p[rowind][columnind]
            if min(p_threshold1,p_threshold2)==0:
                p_threshold1 = max(p_threshold1,p_threshold2)
                p_threshold2 = max(p_threshold1,p_threshold2)
            p_threshold = min(p_threshold1,p_threshold2)
            pR = p_threshold*np.sin(theta)
            pm = pR
            if rescalepR:    
                pm = sum([comb(2,i)*(8*pR/15)**i*(1-8*pR/15)**(2-i)*pR**j*(1-pR)**(1-j) for i in range(3) for j in range(2) if i+j%2])
            if p_threshold>0 and 0<theta<0.499*np.pi and 0<phi<0.499*np.pi and (not truncate or p_threshold*np.cos(theta)*np.sin(phi)*100 < 8):
                pG_list.append(p_threshold*np.cos(theta)*np.cos(phi)*100)
                pT_list.append(p_threshold*np.cos(theta)*np.sin(phi)*100)
                pR_list.append(pm*100)
                pth_list.append(100*np.sqrt((p_threshold*np.cos(theta)*np.cos(phi))**2+(p_threshold*np.cos(theta)*np.sin(phi))**2+pm**2))
            elif p_threshold>0 and (not truncate or p_threshold*np.cos(theta)*np.sin(phi)*100 < 8):
                pG_listB.append(p_threshold*np.cos(theta)*np.cos(phi)*100)
                pT_listB.append(p_threshold*np.cos(theta)*np.sin(phi)*100)
                pR_listB.append(pm*100)
                pth_listB.append(100*np.sqrt((p_threshold*np.cos(theta)*np.cos(phi))**2+(p_threshold*np.cos(theta)*np.sin(phi))**2+pm**2))

    pG_list.extend(pG_listB)
    pT_list.extend(pT_listB)
    pR_list.extend(pR_listB)
    pth_list.extend(pth_listB)

    xi = np.linspace(0, max(pG_list), 200)
    yi = np.linspace(0, max(pT_list), 200)

    if interpolate:
        Zi = griddata((pG_list, pT_list), pR_list, (xi[None, :], yi[:, None]), method='linear')
        Zi[np.isnan(Zi)]=0
        Xi,Yi = np.meshgrid(xi,yi)
        color_dimension = np.sign(Zi)*np.sqrt(Xi**2+Yi**2+Zi**2)
        minn, maxx = min(pth_list), max(pth_list)
        norm=colors.Normalize(minn, maxx,clip=True)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)
        for i in range(len(fcolors)):
            for j in range(len(fcolors)):
                if Zi[i,j]==0:
                    fcolors[i,j] = (1,1,1,1)
                    Zi[i,j] = np.nan
        ax.plot_surface(Xi,Yi,Zi,rstride=1,cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, alpha = alpha, antialiased = False, linewidth=0, shade=False)
        cbar = plt.colorbar(m, ax=ax, shrink = 0.6)
        cbar.set_label('$p_{th}\ [\%]$', rotation=270,labelpad=13.0, fontfamily = 'times')
        # ax.plot_trisurf(pG_list,pT_list,pR_list, alpha= 0.5, cmap=cmap)
        ax.scatter(pG_list,pT_list,pR_list, c=pth_list, cmap=cmap,marker='.')
    else:
        ax.scatter(pG_list,pT_list,pR_list, c=pth_list, cmap=cmap,marker='o', alpha= alpha)
        norm=colors.Normalize(min(pth_list),max(pth_list))
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink = 0.6)
        cbar.set_label('$p_{th}\ [\%]$', rotation=270,labelpad=13.0, fontfamily = 'times')

    
    ax.set_xlabel('$p_G\ [\%]$')
    ax.set_ylabel('$p_T\ [\%]$')
    ax.set_zlabel('$p_{RR}\ [\%]$')

    # ax.set_xticks(np.arange(0,1.8,0.3))
    # ax.set_xlim(0,)
    # ax.set_ylim(0,)
    # ax.set_zlim(1e-5,)
    ax.view_init(elev=20., azim=20)

def fit_plane(LogX_fail_3d_d_p,LogZ_fail_3d_d_p,rescalepR = True):
    theta_phi_pth_list = []
    pG_list = []
    pT_list = []
    pR_list = []
    for rowind in range(len(LogX_fail_3d_d_p)):
        for columnind in range(len(LogX_fail_3d_d_p[rowind])):
            theta, phi, p_threshold1, _ = LogX_fail_3d_d_p[rowind][columnind]
            theta, phi, p_threshold2, _ = LogZ_fail_3d_d_p[rowind][columnind]
            if min(p_threshold1,p_threshold2)==0:
                p_threshold1 = max(p_threshold1,p_threshold2)
                p_threshold2 = max(p_threshold1,p_threshold2)
            p_threshold = min(p_threshold1,p_threshold2)
            pR = p_threshold*np.sin(theta)
            pm = pR
            if rescalepR:    
                pm = sum([comb(2,i)*((8*pR/15)**i)*((1-8*pR/15)**(2-i))*(pR**j)*((1-pR)**(1-j)) for i in range(3) for j in range(2) if i+j%2])
            if p_threshold>0: #and 0<theta<0.499*np.pi and 0<phi<0.499*np.pi:
                pG,pT = [p_threshold*np.cos(theta)*np.cos(phi),p_threshold*np.cos(theta)*np.sin(phi)]
                pG_list.append(pG*100)
                pT_list.append(pT*100)
                pR_list.append(pm*100)
                new_theta = np.arctan2(pm,np.sqrt(pG**2+pT**2))
                new_phi = np.arctan2(pT,pG)
                theta_phi_pth_list.append((new_theta,new_phi,np.sqrt(pG**2+pT**2+pm**2)*100))

    A = np.array(theta_phi_pth_list)

    def func(theta_phi, pGth, pTth, pRth):
        theta,phi = theta_phi.transpose()
        return 1./(np.cos(theta)*np.cos(phi)/pGth + np.cos(theta)*np.sin(phi)/pTth+ np.sin(theta)/pRth)

    guess = (max(pG_list),max(pT_list),max(pR_list))
    params, pcov = optimize.curve_fit(func, A[:,:2], A[:,2], guess)
    fit_error_list = []
    for theta,phi,p_threshold in theta_phi_pth_list:
        if p_threshold>0: #and 0<theta<0.499*np.pi and 0<phi<0.499*np.pi:
            fit_error_list.append(1-func(np.array([theta,phi]),params[0],params[1],params[2])/p_threshold)
    fit_error_list = np.array(fit_error_list)
    return [params,np.sqrt(pcov.diagonal()),np.sqrt(np.mean(fit_error_list**2)),max(fit_error_list)]
