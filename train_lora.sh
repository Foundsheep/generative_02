MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
TIMESTAMP=$(date +%y%m%d_%H%M%S)
echo "Training script for ...$TIMESTAMP"
OUTPUT_DIR="./result_train_$TIMESTAMP"
HUB_MODEL_ID="spr_$TIMESTAMP"
DATASET_NAME="DJMOON/hm_spr_01_04_640_480_partitioned"

accelerate launch diffusers/examples/text_to_image/train_text_to_image_with_new_unet_from_scratch.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --hub_model_id=${HUB_MODEL_ID} \
  --checkpointing_steps=5 \
  --validation_prompts="a simple masking image of self-piercing rivet consisting of 3 plates. The combination consists of rivet type BG0546E, die type DEHG14032, upper plate type SGARC340E with 0.7 mm thickness, middle plate type SGARC440 with 0.7 mm thickness, lower plate type A365.0 with 3.5 mm thickness and its head height from upper plate to rivet's top is 0.7556206201259412 mm" \
  --seed=1105 \
  --push_to_hub \
  --caption_column="caption" \
  --unet_sample_size 480 640 \
  --unet_block_out_channels 32 64 128 256
