#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="A2W"
SOURCE="amazon"
TARGET="webcam"
NOISY_TYPE="ood_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0."
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1


export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="W2A"
SOURCE="webcam"
TARGET="amazon"
NOISY_TYPE="ood_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0."
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="A2D"
SOURCE="amazon"
TARGET="dslr"
NOISY_TYPE="ood_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0."
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE} 
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="D2A"
SOURCE="dslr"
TARGET="amazon"
NOISY_TYPE="ood_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0."
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="W2D"
SOURCE="webcam"
TARGET="dslr"
NOISY_TYPE="ood_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0."
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="D2W"
SOURCE="dslr"
TARGET="webcam"
NOISY_TYPE="ood_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0."
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1



export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="A2W"
SOURCE="amazon"
TARGET="webcam"
NOISY_TYPE="ood_feature" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0.1"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1


export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="W2A"
SOURCE="webcam"
TARGET="amazon"
NOISY_TYPE="ood_feature" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0.1"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="A2D"
SOURCE="amazon"
TARGET="dslr"
NOISY_TYPE="ood_feature" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0.1"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE} 
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="D2A"
SOURCE="dslr"
TARGET="amazon"
NOISY_TYPE="ood_feature" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0.1"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="W2D"
SOURCE="webcam"
TARGET="dslr"
NOISY_TYPE="ood_feature" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0.1"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V1"
PROJ_NAME="D2W"
SOURCE="dslr"
TARGET="webcam"
NOISY_TYPE="ood_feature" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0.1"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1