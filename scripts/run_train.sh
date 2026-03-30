# export MODEL_DIR="stabilityai/stable-diffusion-2-1-base"
export MODEL_DIR="Manojb/stable-diffusion-2-1-base"
export OUTPUT_DIR="/workspace/results/DeDistortNet"
export TRAIN_JSON_FILE="/workspace/data/preprocessed/PROSTATEx_train_metadata.jsonl"
export VAL_JSON_FILE="/workspace/data/preprocessed/PROSTATEx_validation_metadata.jsonl"
export PLOT_JSON_FILE="/workspace/data/preprocessed/PROSTATEx_plot_metadata.jsonl"
export DATA_DIR="/workspace/data/preprocessed/"
export IMAGE_ENCODER_PATH="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

cd /workspace/source

CUDA_VISIBLE_DEVICES="0" accelerate launch train_DeDistortNet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --data_json_file=$TRAIN_JSON_FILE \
 --val_data_json_file=$VAL_JSON_FILE \
 --plot_data_json_file=$PLOT_JSON_FILE \
 --data_root_path=$DATA_DIR \
 --image_encoder_path=$IMAGE_ENCODER_PATH \
 --resolution=512 \
 --displacement_rate=32 \
 --random_y_squeeze_rate=0.1 \
 --random_rotation_degree=15 \
 --random_shift_range=4 \
 --image_condition_drop_rate=0.0 \
 --num_train_epochs=100000 \
 --learning_rate=1e-5 \
 --weight_decay=1e-6 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --dataloader_num_workers=24 \
 --checkpointing_steps=10000 \
 --mixed_precision="fp16" \
 --seed=42 \
