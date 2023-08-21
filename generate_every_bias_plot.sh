#!/bin/bash

python3 generate_Gbias_plot.py code=SC3 logical=X dmin=11 dmax=17 Tbias=20 Num_p=50 pthguess=0.03 delpth=1. max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;
python3 generate_Gbias_plot.py code=SC3 logical=Z dmin=11 dmax=17 Tbias=20 Num_p=50 pthguess=0.03 delpth=1. max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;

python3 generate_Gbias_plot.py code=RSC logical=X dmin=11 dmax=17 Tbias=20 Num_p=60 pthguess=0.04 delpth=1. max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;
python3 generate_Gbias_plot.py code=RSC logical=Z dmin=11 dmax=17 Tbias=20 Num_p=60 pthguess=0.04 delpth=1. max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;

python3 generate_Gbias_plot.py code=XZZX logical=X dmin=11 dmax=17 Tbias=20 Num_p=60 pthguess=0.006 delpth=1. max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;
python3 generate_Gbias_plot.py code=XZZX logical=Z dmin=11 dmax=17 Tbias=20 Num_p=60 pthguess=0.006 delpth=1. max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;

python3 generate_Gbias_plot.py code=XYZ2 logical=Y dmin=11 dmax=17 Tbias=20 Num_p=60 pthguess=0.004 delpth=0.5 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;
python3 generate_Gbias_plot.py code=XYZ2 logical=Z dmin=11 dmax=17 Tbias=20 Num_p=60 pthguess=0.004 delpth=0.5 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.4;

python3 generate_Gbias_plot.py code=HFC logical=X dmin=4 dmax=8 Tbias=20 Num_p=60 pthguess=0.005 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45;
python3 generate_Gbias_plot.py code=HFC logical=Z dmin=4 dmax=8 Tbias=20 Num_p=60 pthguess=0.005 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45;

python3 generate_Gbias_plot.py code=FCC logical=X dmin=4 dmax=8 Tbias=20 Num_p=60 pthguess=0.005 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45;
python3 generate_Gbias_plot.py code=FCC logical=Z dmin=4 dmax=8 Tbias=20 Num_p=60 pthguess=0.005 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45;



python3 generate_Tbias_plot.py code=SC3 logical=X dmin=11 dmax=17 Gbias=1 Num_p=30 pthguess=0.1 delpth=0.3 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.49 T_over_d=0.5;
python3 generate_Tbias_plot.py code=SC3 logical=Z dmin=11 dmax=17 Gbias=1 Num_p=30 pthguess=0.1 delpth=0.3 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.49 T_over_d=0.5;

python3 generate_Tbias_plot.py code=XZZX logical=X dmin=11 dmax=17 Gbias=1 Num_p=60 pthguess=0.17 delpth=0.8 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.49 T_over_d=1;
python3 generate_Tbias_plot.py code=XZZX logical=Z dmin=11 dmax=17 Gbias=1 Num_p=60 pthguess=0.17 delpth=0.8 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;

python3 generate_Tbias_plot.py code=RSC logical=X dmin=11 dmax=17 Gbias=1 Num_p=40 pthguess=0.1 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;
python3 generate_Tbias_plot.py code=RSC logical=Z dmin=11 dmax=17 Gbias=1 Num_p=40 pthguess=0.1 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;

python3 generate_Tbias_plot.py code=XYZ2 logical=Y dmin=11 dmax=17 Gbias=1 Num_p=40 pthguess=0.035 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;
python3 generate_Tbias_plot.py code=XYZ2 logical=Z dmin=11 dmax=17 Gbias=1 Num_p=40 pthguess=0.035 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;

python3 generate_Tbias_plot.py code=HFC logical=X dmin=4 dmax=8 Gbias=1 Num_p=60 pthguess=0.013 delpth=0.6 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;
python3 generate_Tbias_plot.py code=HFC logical=Z dmin=4 dmax=8 Gbias=1 Num_p=60 pthguess=0.013 delpth=0.6 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;

python3 generate_Tbias_plot.py code=FCC logical=X dmin=4 dmax=8 Gbias=1 Num_p=60 pthguess=0.008 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;
python3 generate_Tbias_plot.py code=FCC logical=Z dmin=4 dmax=8 Gbias=1 Num_p=60 pthguess=0.008 delpth=0.4 max_shots=300_000 shot_batch=3_000 max_num_fail=30_000 max_fail_rate=0.45 T_over_d=1;