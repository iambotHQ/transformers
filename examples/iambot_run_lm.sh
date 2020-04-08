#!bash -e

OUTPUT_DIR=service_models/vanilla/roberta-pl-misspelled
DATA_FILE=data/data/pl/lm/pl.txt
MODEL_DIR=${OUTPUT_DIR}/model

mkdir -pv ${OUTPUT_DIR}

if [[ ! -f ${OUTPUT_DIR}/sp.model ]]; then
    spm_train --input ${DATA_FILE} \
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
        --vocab_size 25000 \
        --character_coverage 1.0 \
        --model_type "bpe" \
        --num_threads $(nproc) \
        --remove_extra_whitespaces true \
        --shuffle_input_sentence true \
        --user_defined_symbols "<cls>,<mask>" \
        --character_coverage 1.0

    rm -v sp.vocab
    mv -v sp.model ${OUTPUT_DIR}/sp.model
fi
python -m torch.distributed.launch --nproc_per_node=16 \
    transformers/examples/run_iambot_language_modeling.py \
    --output_dir $MODEL_DIR \
    --model_type roberta \
    --mlm \
    --config_name $OUTPUT_DIR \
    --tokenizer_name $OUTPUT_DIR \
    --train_data_file ${OUTPUT_DIR}/train.txt \
    --eval_data_file ${OUTPUT_DIR}/eval.txt \
    --line_by_line \
    --do_train \
    --num_train_epochs 15 \
    --per_gpu_train_batch_size 8 \
    --overwrite_output_dir \
    --iambot \
    --iambot_all_data ${DATA_FILE} \
    --iambot_save_every_epoch \
    --iambot_transform_ratio 0.25 \
    --iambot_misspell_prob 0.25 \
    --iambot_uppercase_prob 0.25 \
    --iambot_lowercase_prob 0.25 \
    --iambot_remove_char_prob 0.25 \
    --iambot_train_eval_ratio 0.03
