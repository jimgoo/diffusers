# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

## 768x768 (all solid color samples)
# export MODEL_NAME="stabilityai/stable-diffusion-2"

## 512x512
export MODEL_NAME="stabilityai/stable-diffusion-2-base"

export OUTPUT_DIR="../../../models/alvan_shivam_v2-base"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=400 \
  --save_sample_prompt="photo of zwx dog" \
  --concepts_list="concepts_list.json"

## orig 
#  2.38it/s, loss=0.254, lr=1e-6
## v2-base
#  2.92it/s, loss=0.269, lr=1e-6

# /home/jimgoo/git/art/diffusers-shiv/src/diffusers/utils/deprecation_utils.py:35: FutureWarning: The configuration file of this scheduler: DDIMScheduler {
#   "_class_name": "DDIMScheduler",
#   "_diffusers_version": "0.9.0",
#   "beta_end": 0.012,
#   "beta_schedule": "scaled_linear",
#   "beta_start": 0.00085,
#   "clip_sample": false,
#   "num_train_timesteps": 1000,
#   "prediction_type": "epsilon",
#   "set_alpha_to_one": false,
#   "steps_offset": 0,
#   "trained_betas": null
# }
#  is outdated. `steps_offset` should be set to 1 instead of 0.