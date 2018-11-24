#!/usr/bin/env bash

# This script is used to setup the riai virtual machine by copying the python scripts from this repo to the correct
# directories in the riai virtual machine.

cp src/analyzer/*.py /home/riai2018/analyzer/
cp src/analyzer/test.sh /home/riai2018/analyzer/
cp gurobi.lic /home/riai2018/gurobi.lic
