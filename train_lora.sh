MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="./result_train_lora"
HUB_MODEL_ID="lora_01"
DATASET_NAME="DJMOON/hm_spr_01_03_640_480"

accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --checkpointing_steps=500 \
  --validation_prompt="SPR2_BG0G46E_DEHG13598_SABC1470(1.1t)_A365.0(3.0t)_0.640493" \
  --seed=1337
  --caption_column="caption"
