#!/bin/bash

Nphi=20
Ncores=20
((Nlim=$Nphi*$Nphi/2+$Nphi/2))
for ((i=0;i<=Nlim;i+=$Ncores))
do

python3 generate_3d_plot.py code=XYZ2 logical=Y dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.0046 pTmax=0.042 pRmax=0.075 Num_p=30 delpth=0.3 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4 Nphi=$Nphi Nstart=$i;                                    
python3 generate_3d_plot.py code=XYZ2 logical=Z dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.0046 pTmax=0.042 pRmax=0.075 Num_p=30 delpth=0.3 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4 Nphi=$Nphi Nstart=$i;

python3 generate_3d_plot.py code=XZZX logical=X dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.005 pTmax=0.18 pRmax=0.07 Num_p=30 delpth=0.5 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 Nphi=$Nphi Nstart=$i;
python3 generate_3d_plot.py code=XZZX logical=Z dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.005 pTmax=0.18 pRmax=0.07 Num_p=30 delpth=0.5 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 Nphi=$Nphi Nstart=$i;

python3 generate_3d_plot.py code=SC3 logical=X dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.007 pTmax=0.06 pRmax=0.04 Num_p=30 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.49 Nphi=$Nphi Nstart=$i;
python3 generate_3d_plot.py code=SC3 logical=Z dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.007 pTmax=0.06 pRmax=0.04 Num_p=30 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.49 Nphi=$Nphi Nstart=$i;

python3 generate_3d_plot.py code=RSC logical=X dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.007 pTmax=0.06 pRmax=0.07 Num_p=30 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4 Nphi=$Nphi Nstart=$i;
python3 generate_3d_plot.py code=RSC logical=Z dmin=11 dmax=17 Tbias=20 Gbias=1 pGmax=0.007 pTmax=0.06 pRmax=0.07 Num_p=30 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4 Nphi=$Nphi Nstart=$i;

python3 generate_3d_plot.py code=FCC logical=X dmin=5 dmax=8 Tbias=20 Gbias=1 pGmax=0.005 pTmax=0.0065 pRmax=0.0075 Num_p=30 delpth=0.2 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 Nphi=$Nphi Nstart=$i;
python3 generate_3d_plot.py code=FCC logical=Z dmin=5 dmax=8 Tbias=20 Gbias=1 pGmax=0.005 pTmax=0.0065 pRmax=0.0075 Num_p=30 delpth=0.2 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 Nphi=$Nphi Nstart=$i;

python3 generate_3d_plot.py code=HFC logical=X dmin=5 dmax=8 Tbias=20 Gbias=1 pGmax=0.0042 pTmax=0.012 pRmax=0.005 Num_p=30 delpth=0.2 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 Nphi=$Nphi Nstart=$i;
python3 generate_3d_plot.py code=HFC logical=Z dmin=5 dmax=8 Tbias=20 Gbias=1 pGmax=0.0042 pTmax=0.012 pRmax=0.005 Num_p=30 delpth=0.2 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 Nphi=$Nphi Nstart=$i;

done

python3 merge_threshold_output.py
