#!/bin/bash

model_list=(cold  cold-expose  cold-softmax  ecm  ecmm esmm  fscd  fscd-expose  fscd-softmax  vector-product)

for one in ${model_list[@]}
do
    echo $one
    /usr/local/anaconda3/bin/python src/start_train_cmd.py $one > nohup_${one}.out 2>&1
    wait
done
