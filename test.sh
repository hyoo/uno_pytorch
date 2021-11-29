#!/bin/bash

# AUC prediction model
if [ ! -f "top_21_auc_1fold.uno.h5" ]; then
  curl -o top_21_auc_1fold.uno.h5 http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
fi

echo "Classification model"
CMD="python uno_pytorch.py --device=cuda -z 512 -lr 4e-4 --mode cls"
echo $CMD
$CMD

echo "Regression model"
CMD="python uno_pytorch.py --device=cuda -z 512 -lr 4e-4 --mode reg"
echo $CMD
$CMD