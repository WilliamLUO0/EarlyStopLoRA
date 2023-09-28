# Modified by xxxxx on 25/9/2023, based on the script by @Akegarasu

# Path & Data Settings
pretrained_model="./models/Stable-diffusion/xxxxx.safetensors"  # just for an example
train_data_dir="./train/image_xxx"
reg_data_dir=""
is_v2_model=0
parameterization=0

# LoRA settings
network_module="networks.lora"
network_weights=""
network_dim=128
network_alpha=64

# Train Parameters
resolution="512,512"
batch_size=1
max_train_epoches=15
save_every_n_epochs=1
train_unet_only=0
train_text_encoder_only=0
stop_text_encoder_training=0
noise_offset="0"
keep_tokens=0
min_snr_gamma=0
lr="1e-05"
unet_lr="1e-05"
text_encoder_lr="1e-06"
lr_scheduler="constant"  # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"
lr_warmup_steps=0
lr_restart_cycles=1
optimizer_type="AdamW8bit"  # Optimizer type: AdamW AdamW8bit Lion SGDNesterov SGDNesterov8bit DAdaptation AdaFactor

# Output & Resume
output_name="test_"
save_model_as="safetensors"  # model save ext | 模型保存格式 ckpt, pt, safetensors
save_state=0
resume=""

# Other Settings
min_bucket_reso=256
max_bucket_reso=1024
persistent_data_loader_workers=0
clip_skip=2
multi_gpu=0
lowram=0

# LyCORIS Training Settings
algo="lora"
conv_dim=4
conv_alpha=4
dropout="0"

# Remote logging settings
use_wandb=0
wandb_api_key=""
log_tracker_name=""

export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

extArgs=()
launchArgs=()

[ $multi_gpu -eq 1 ] && launchArgs+=("--multi_gpu")
[ $is_v2_model -eq 1 ] && extArgs+=("--v2") || extArgs+=("--clip_skip $clip_skip")
[ $parameterization -eq 1 ] && extArgs+=("--v_parameterization")
[ $train_unet_only -eq 1 ] && extArgs+=("--network_train_unet_only")
[ $train_text_encoder_only -eq 1 ] && extArgs+=("--network_train_text_encoder_only")
[ -n "$network_weights" ] && extArgs+=("--network_weights $network_weights")
[ -n "$reg_data_dir" ] && extArgs+=("--reg_data_dir $reg_data_dir")
[ -n "$optimizer_type" ] && extArgs+=("--optimizer_type $optimizer_type")
[ $optimizer_type == "DAdaptation" ] && extArgs+=("--optimizer_args decouple=True")
[ $save_state -eq 1 ] && extArgs+=("--save_state")
[ -n "$resume" ] && extArgs+=("--resume $resume")
[ $persistent_data_loader_workers -eq 1 ] && extArgs+=("--persistent_data_loader_workers")
[ $network_module == "lycoris.kohya" ] && extArgs+=("--network_args conv_dim=$conv_dim conv_alpha=$conv_alpha algo=$algo dropout=$dropout")
[ $stop_text_encoder_training -ne 0 ] && extArgs+=("--stop_text_encoder_training $stop_text_encoder_training")
[ $noise_offset != "0" ] && extArgs+=("--noise_offset $noise_offset")
[ $min_snr_gamma -ne 0 ] && extArgs+=("--min_snr_gamma $min_snr_gamma")
[ $use_wandb -eq 1 ] && extArgs+=("--log_with=all") || extArgs+=("--log_with=tensorboard")
[ -n "$wandb_api_key" ] && extArgs+=("--wandb_api_key $wandb_api_key")
[ -n "$log_tracker_name" ] && extArgs+=("--log_tracker_name $log_tracker_name")
[ $lowram ] && extArgs+=("--lowram")

python -m accelerate.commands.launch ${launchArgs[@]} --num_cpu_threads_per_process=8 "./sd-scripts/es_lora_ldm.py" \
  --enable_bucket \
  --pretrained_model_name_or_path=$pretrained_model \
  --train_data_dir=$train_data_dir \
  --output_dir="./output" \
  --logging_dir="./logs" \
  --log_prefix=$output_name \
  --resolution=$resolution \
  --network_module=$network_module \
  --max_train_epochs=$max_train_epoches \
  --learning_rate=$lr \
  --unet_lr=$unet_lr \
  --text_encoder_lr=$text_encoder_lr \
  --lr_scheduler=$lr_scheduler \
  --lr_warmup_steps=$lr_warmup_steps \
  --lr_scheduler_num_cycles=$lr_restart_cycles \
  --network_dim=$network_dim \
  --network_alpha=$network_alpha \
  --output_name=$output_name \
  --train_batch_size=$batch_size \
  --save_every_n_epochs=$save_every_n_epochs \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --seed="1337" \
  --cache_latents \
  --prior_loss_weight=1 \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as=$save_model_as \
  --min_bucket_reso=$min_bucket_reso \
  --max_bucket_reso=$max_bucket_reso \
  --keep_tokens=$keep_tokens \
  --xformers --shuffle_caption ${extArgs[@]} \
  --calculate_cs \
  --cs_interval 20 \
  --window_size 50