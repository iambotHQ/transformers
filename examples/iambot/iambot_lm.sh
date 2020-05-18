#!bash -e

DATA_ROOT=res/raw_data/language_modeling/pl

OUTPUT_DIR=service_models/vanilla/roberta-pl-misspelled
DATA_FILE=$DATA_ROOT/all.txt
MODEL_DIR=$OUTPUT_DIR/model
TRAIN_SCRIPT=transformers/examples/iambot/train_lm.py
LOGGING_DIR=$MODEL_DIR/logs/train

if [[ ! -f $DATA_FILE ]]; then
    echo "Merging data"
    cat $DATA_ROOT/wiki.txt $DATA_ROOT/opensubs.txt >$DATA_FILE
fi

mkdir -vpv $LOGGING_DIR
mkdir -vpv $OUTPUT_DIR

if [[ ! -f $OUTPUT_DIR/sp.model ]]; then
    spm_train --input $DATA_FILE \
        --accept_language "pl," \
        --bos_piece "<s>" \
        --bos_id 0 \
        --eos_piece "</s>" \
        --eos_id 1 \
        --pad_piece "<pad>" \
        --pad_id 2 \
        --unk_piece "<unk>" \
        --unk_id 3 \
        --input_format "text" \
        --model_prefix "sp" \
        --vocab_size 20000 \
        --model_type "unigram" \
        --num_threads $(nproc) \
        --remove_extra_whitespaces true \
        --shuffle_input_sentence true \
        --input_sentence_size 50000000 \
        --user_defined_symbols "<cls>,<mask>,<sep>" \
        --character_coverage 1.0

    rm -v sp.vocab
    mv -v sp.model $OUTPUT_DIR/sp.model
fi

if [[ $1 == 'eval' ]]; then
    CUDA_VISIBLE_DEVICES=0, python $TRAIN_SCRIPT \
        --output_dir $MODEL_DIR \
        --model_type roberta \
        --model_name_or_path $OUTPUT_DIR \
        --logging_dir $LOGGING_DIR \
        --mlm \
        --train_data_file $OUTPUT_DIR/train.txt \
        --eval_data_file $OUTPUT_DIR/eval.txt \
        --line_by_line \
        --do_eval_all \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 12 \
        --iambot_mode \
        --iambot_all_data $DATA_FILE \
        --iambot_transforms
elif [[ $1 == 'train' ]]; then
    if [[ $2 == 'misspelled' ]]; then
        python -m torch.distributed.launch --nproc_per_node=16 $TRAIN_SCRIPT \
            --output_dir $MODEL_DIR \
            --model_type roberta \
            --config_name $OUTPUT_DIR \
            --tokenizer_name $OUTPUT_DIR \
            --logging_dir $LOGGING_DIR \
            --mlm \
            --train_data_file $OUTPUT_DIR/train.txt \
            --eval_data_file $OUTPUT_DIR/eval.txt \
            --line_by_line \
            --save_steps 1000 \
            --logging_steps 100 \
            --do_train \
            --num_train_epochs 15 \
            --per_gpu_train_batch_size 8 \
            --overwrite_output_dir \
            --iambot_mode \
            --iambot_all_data $DATA_FILE \
            --iambot_tokenizer_sampling \
            --iambot_transforms \
            --iambot_transform_ratio 0.25 \
            --iambot_misspell_prob 0.25 \
            --iambot_uppercase_prob 0.25 \
            --iambot_lowercase_prob 0.25 \
            --iambot_remove_char_prob 0.25 \
            --iambot_train_eval_ratio 0.03
    elif [[ $2 == 'default' ]]; then
        python -m torch.distributed.launch --nproc_per_node=16 $TRAIN_SCRIPT \
            --output_dir $MODEL_DIR \
            --model_type roberta \
            --mlm \
            --config_name $OUTPUT_DIR \
            --tokenizer_name $OUTPUT_DIR \
            --train_data_file $OUTPUT_DIR/train.txt \
            --eval_data_file $OUTPUT_DIR/eval.txt \
            --line_by_line \
            --save_steps 1000 \
            --logging_steps 50 \
            --do_train \
            --num_train_epochs 15 \
            --per_gpu_train_batch_size 8 \
            --overwrite_output_dir \
            --iambot \
            --iambot_all_data ${DATA_FILE}
    else
        echo "Use 'default' or 'misspelled' mode"
    fi
else
    echo "Use 'train' or 'eval' mode"
fi
