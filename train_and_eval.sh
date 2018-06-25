#!/bin/bash

if [ $1 ]; then
  gpu=$1
else
  gpu="0"
fi

if [ $2 ]; then
  result_path=$2
else
  result_path="result.txt"
fi

if [ $3 ]; then
  dims=$3
else
  dims="32_16"
fi

# train
python3 main.py $gpu model.h5 $dims

# validation
python3 valid.py $gpu model.h5 predict.txt

# evaluation
perl semeval2010_task8_scorer-v1.2.pl predict.txt data/answer_key.txt > $result_path
