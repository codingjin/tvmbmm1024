#!/bin/bash
set -x

python lastbatchmatmul.py 2>&1 | tee lastbatchmatmul.log

python measure_energy.py --funcso=top.so 2>&1 | tee measure_energy_top.log

python measure_energy.py --funcso=last.so 2>&1 | tee measure_energy_last.log

python measure_perf.py --funcso=top.so 2>&1 | tee measure_perf_top.log

python measure_perf.py --funcso=last.so 2>&1 | tee measure_perf_last.log

python measuretorch_energy.py 2>&1 | tee measuretorch_energy.log

python measuretorch_perf.py 2>&1 | tee measuretorch_perf.log
