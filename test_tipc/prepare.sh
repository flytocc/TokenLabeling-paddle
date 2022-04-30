#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_lite_infer']

MODE=$2

dataline=$(cat ${FILENAME})
# parser params
IFS=$'\n'
lines=(${dataline})

function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    if [ ${#array[*]} = 2 ]; then
        echo ${array[1]}
    else
    	IFS="|"
    	tmp="${array[1]}:${array[2]}"
        echo ${tmp}
    fi
}

function func_get_url_file_name(){
    strs=$1
    IFS="/"
    array=(${strs})
    tmp=${array[${#array[@]}-1]}
    echo ${tmp}
}

model_name=$(func_parser_value "${lines[1]}")

model_name=$(func_parser_value "${lines[1]}")
model_url_value=$(func_parser_value "${lines[35]}")
model_url_key=$(func_parser_key "${lines[35]}")

if [ ${MODE} = "lite_train_lite_infer" ];then
    # pretrain lite train data
    mkdir dataset
    cd dataset
    rm -rf ILSVRC2012
    tar xf whole_chain_little_train.tar
    ln -s whole_chain_little_train ILSVRC2012
    cd ILSVRC2012 
    mv train.txt train_list.txt
    mv val.txt val_list.txt
    cp -r train/* val/
    cd ../../
fi
