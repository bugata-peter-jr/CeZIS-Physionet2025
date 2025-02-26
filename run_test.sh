#!/usr/bin/bash

nohup python run_model.py -d /projects/ECG/data/CODE-15/wfdb_chagas_full -m ./$1 -o ./$2 -v >$3 2>&1 &


