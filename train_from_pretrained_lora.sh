MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
TIMESTAMP=$(date +%y%m%d_%H%M%S)
echo "Training script for ...$TIMESTAMP"
OUTPUT_DIR="./result_train_lora_$TIMESTAMP"
HUB_MODEL_ID="spr_lora_$TIMESTAMP"
DATASET_NAME="DJMOON/hm_spr_01_04_640_480_partitioned"

accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --random_flip \
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
  --seed=1337 \
  --caption_column="caption" \
#   --push_to_hub \
#   --validation_prompt="a simple masking image of self-piercing rivet consisting of 2 plates. The combination consists of rivet type BG0G46E, die type DEHG13598, upper plate type SABC1470 with 1.1 mm thickness, lower plate type A365.0 with 3.0 mm thickness and its head height from upper plate to rivet's top is 0.8395354566597028 mm" \
#   --num_validation_images=1