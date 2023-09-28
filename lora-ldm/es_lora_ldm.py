# Modified by xxxxx on 25/9/2023, based on the script by @kohya-ss

import importlib
import argparse
import gc
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
import toml
from copy import deepcopy
import numpy as np
import math
from collections import deque


from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import model_util

import library.train_util as train_util
from library.train_util import (
    DreamBoothDataset,
)
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
)


class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    def generate_step_logs(
        self, args: argparse.Namespace, current_loss, avr_loss, lr_scheduler, keys_scaled=None, mean_norm=None, maximum_norm=None
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()

        if args.network_train_text_encoder_only or len(lrs) <= 2:
            if args.network_train_unet_only:
                logs["lr/unet"] = float(lrs[0])
            elif args.network_train_text_encoder_only:
                logs["lr/textencoder"] = float(lrs[0])
            else:
                logs["lr/textencoder"] = float(lrs[0])
                logs["lr/unet"] = float(lrs[-1])

            if (
                args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
            ):
                logs["lr/d*lr"] = (
                    lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        pass

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype
    ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0], weight_dtype)
        return encoder_hidden_states

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noise_pred = unet(noisy_latents, timesteps, text_conds).sample
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            if use_user_config:
                print(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    print(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    print("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    print("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group)

        print("preparing accelerator")
        accelerator, logging_dir = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            accelerator.wait_for_everyone()

        self.cache_text_encoder_outputs_if_needed(
            args, accelerator, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype
        )

        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            print(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()

        accelerator.print("prepare optimizer, data loader etc.")

        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            accelerator.print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)"
            )
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collater,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        if args.full_fp16:
            assert (
                    args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                    args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)

        if train_unet and train_text_encoder:
            if len(text_encoders) > 1:
                unet, t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoders = [text_encoder]
        elif train_unet:
            unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, network, optimizer, train_dataloader, lr_scheduler
            )
        elif train_text_encoder:
            if len(text_encoders) > 1:
                t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    text_encoder, network, optimizer, train_dataloader, lr_scheduler
                )
                text_encoders = [text_encoder]

            unet.to(accelerator.device, dtype=weight_dtype)
        else:
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )

        text_encoders = train_util.transform_models_if_DDP(text_encoders)
        unet, network = train_util.transform_models_if_DDP([unet, network])

        if args.gradient_checkpointing:
            print("gradient_checkpointing: True")
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()

                t_enc.text_model.embeddings.requires_grad_(True)
        else:
            print("gradient_checkpointing: False")
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        network.prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
        }

        if use_user_config:
            datasets_metadata = []
            tag_frequency = {}
            dataset_dirs_info = {}

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file

                    subsets_metadata.append(subset_metadata)

                    if image_dir_or_metadata_file is not None:
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs
            )

        loss_list = []
        loss_total = 0.0
        del train_dataset_group

        if hasattr(network, "on_step_start"):
            on_step_start = network.on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        def moving_average(values, window_size):
            return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

        def moving_variance(values, window_size):
            return [np.var(values[i:i + window_size]) for i in range(len(values) - window_size + 1)]

        def difference_window(values):
            return np.diff(values)

        def _compute_cosine_similarity(x1, x2, x3):
            if len(x2.shape) == 4 and len(x3.shape) == 4:
                if x2.shape[-2:] != (1, 1) or x3.shape[-2:] != (1, 1):
                    print(x2.shape)
                    print(x3.shape)
                    raise ValueError("1*1的卷积啦")

            x2 = torch.squeeze(x2)
            x3 = torch.squeeze(x3)
            y = torch.matmul(x2, x3)

            flattened_cosine_similarity = torch.nn.functional.cosine_similarity(x1.view(1, -1), y.view(1, -1))

            del x2, x3, y
            torch.cuda.empty_cache()

            return flattened_cosine_similarity

        def compute_cosine_similarity(device, unet, text_encoder, lora_weight):
            text_encoder_results = []
            unet_results = []
            if unet is not None:
                lora_unet_dict = {k.replace('lora_unet_', ''): v.to(device) for k, v in lora_weight.items() if
                                  k.startswith('lora_unet')}
                unet = {k.replace('.', '_'): v.to(device) for k, v in unet.state_dict().items()}
                # print(len(lora_unet_dict))  # 576
                # print(len(unet))  # 686
                for k in [k for k in lora_unet_dict.keys() if '.lora_down.weight' in k]:
                    prefix = k.split('.')[0]
                    if f"{prefix}_weight" in unet.keys() and f"{prefix}.lora_down.weight" in lora_unet_dict.keys() and f"{prefix}.lora_up.weight" in lora_unet_dict.keys():
                        result = _compute_cosine_similarity(unet[f"{prefix}_weight"],
                                                            lora_unet_dict[f"{prefix}.lora_up.weight"],
                                                            lora_unet_dict[f"{prefix}.lora_down.weight"])
                        unet_results.append(result)
                # print(len(unet_results))  # 192

            if text_encoder is not None:
                lora_te_dict = {k.replace('lora_te_', ''): v.to(device) for k, v in lora_weight.items() if
                                k.startswith('lora_te')}
                text_encoder = {k.replace('.', '_'): v.to(device) for k, v in text_encoder.state_dict().items()}
                # print(len(lora_te_dict))  # 216
                # print(len(text_encoder))  # 197
                for k in [k for k in lora_te_dict.keys() if '.lora_down.weight' in k]:
                    prefix = k.split('.')[0]
                    if f"{prefix}_weight" in text_encoder.keys() and f"{prefix}.lora_down.weight" in lora_te_dict.keys() and f"{prefix}.lora_up.weight" in lora_te_dict.keys():
                        # print(prefix)
                        result = _compute_cosine_similarity(text_encoder[f"{prefix}_weight"],
                                                            lora_te_dict[f"{prefix}.lora_up.weight"],
                                                            lora_te_dict[f"{prefix}.lora_down.weight"])
                        text_encoder_results.append(result)
                # print(len(text_encoder_results))  # 72

            text_encoder_results = np.array([tensor.cpu().numpy() for tensor in text_encoder_results]).reshape(1, -1)
            unet_results = np.array([tensor.cpu().numpy() for tensor in unet_results]).reshape(1, -1)

            return text_encoder_results, unet_results

        def compute_cosine_similarity_tensor(device, unet, text_encoder, lora_weight):
            text_encoder_results = []
            unet_results = []
            if unet is not None:
                lora_unet_dict = {k.replace('lora_unet_', ''): v.to(device) for k, v in lora_weight.items() if
                                  k.startswith('lora_unet')}
                unet = {k.replace('.', '_'): v.to(device) for k, v in unet.state_dict().items()}
                for k in [k for k in lora_unet_dict.keys() if '.lora_down.weight' in k]:
                    prefix = k.split('.')[0]
                    if f"{prefix}_weight" in unet.keys() and f"{prefix}.lora_down.weight" in lora_unet_dict.keys() and f"{prefix}.lora_up.weight" in lora_unet_dict.keys():
                        result = _compute_cosine_similarity(unet[f"{prefix}_weight"],
                                                            lora_unet_dict[f"{prefix}.lora_up.weight"],
                                                            lora_unet_dict[f"{prefix}.lora_down.weight"])
                        unet_results.append(result)

            if text_encoder is not None:
                lora_te_dict = {k.replace('lora_te_', ''): v.to(device) for k, v in lora_weight.items() if
                                k.startswith('lora_te')}
                text_encoder = {k.replace('.', '_'): v.to(device) for k, v in text_encoder.state_dict().items()}
                for k in [k for k in lora_te_dict.keys() if '.lora_down.weight' in k]:
                    prefix = k.split('.')[0]
                    if f"{prefix}_weight" in text_encoder.keys() and f"{prefix}.lora_down.weight" in lora_te_dict.keys() and f"{prefix}.lora_up.weight" in lora_te_dict.keys():
                        result = _compute_cosine_similarity(text_encoder[f"{prefix}_weight"],
                                                            lora_te_dict[f"{prefix}.lora_up.weight"],
                                                            lora_te_dict[f"{prefix}.lora_down.weight"])
                        text_encoder_results.append(result)

            text_encoder_results = torch.stack(text_encoder_results).to(device)
            unet_results = torch.stack(unet_results).to(device)

            return text_encoder_results, unet_results

        training_cs_data = []
        cs_interval = args.cs_interval
        window_size = args.window_size
        cs_values_list = []
        ma_cs_values_list = []
        diff_values_list = []
        ma_diff_values_list = []
        var_values_list = []
        prev_derivative = None
        valley_count = 0
        early_stop = False
        save_at_tp = False
        weights_queue = deque(maxlen=(3*window_size-1))

        # training loop

        from torch.utils.tensorboard import SummaryWriter
        logging_dir = os.path.join(logging_dir, "network_train")
        writer = SummaryWriter(logging_dir)
        global_step_counter = 0

        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            network.on_epoch_start(text_encoder, unet)

            for step, batch in enumerate(train_dataloader):

                global_step_counter += 1
                current_step.value = global_step
                with accelerator.accumulate(network):
                    on_step_start(text_encoder, unet)

                    with torch.no_grad():
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(accelerator.device)
                        else:
                            latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample()

                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                        latents = latents * self.vae_scale_factor
                    b_size = latents.shape[0]

                    with torch.set_grad_enabled(train_text_encoder):
                        if args.weighted_captions:
                            text_encoder_conds = get_weighted_text_embeddings(
                                tokenizer,
                                text_encoder,
                                batch["captions"],
                                accelerator.device,
                                args.max_token_length // 75 if args.max_token_length else 1,
                                clip_skip=args.clip_skip,
                            )
                        else:
                            text_encoder_conds = self.get_text_cond(
                                args, accelerator, batch, tokenizers, text_encoders, weight_dtype
                            )

                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents
                    )

                    with accelerator.autocast():
                        noise_pred = self.call_unet(
                            args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype
                        )

                    if args.v_parameterization:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])

                    loss_weights = batch["loss_weights"]
                    loss = loss * loss_weights

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)

                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = network.get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    if args.calculate_cs:
                        if global_step_counter % cs_interval == 0:
                            te_result, unet_result = compute_cosine_similarity_tensor(accelerator.device, unet, text_encoder, dict(network.named_parameters()))
                            cs = unet_result.mean()
                            cs = cs.detach().cpu().item()
                            te = te_result.mean()
                            te = te.detach().cpu().item()
                            result = np.vstack((te_result.detach().cpu().numpy(), unet_result.detach().cpu().numpy()))
                            cs_to = np.mean(result)
                            writer.add_scalar('unet_cs/step', cs, global_step_counter)
                            writer.add_scalar('te_cs/step', te, global_step_counter)
                            writer.add_scalar('total_cs/step', cs_to, global_step_counter)

                            def convert_float32(value):
                                if isinstance(value, np.float32):
                                    return float(value)
                                return value

                            training_cs_data.append({
                                'step': global_step_counter,
                                'unet_cs': convert_float32(cs),
                                'te_cs': convert_float32(te),
                                'total_cs': convert_float32(cs_to),
                            })

                            current_weights = {'epoch': epoch, 'global_step': global_step, 'network_state': accelerator.unwrap_model(network), "global_step_counter": global_step_counter}
                            weights_queue.append(current_weights)

                            cs_values_list.append(cs)
                            if len(cs_values_list) >= window_size:
                                ma_cs_values = moving_average(cs_values_list, window_size)
                                ma_cs_values_list.append(ma_cs_values[-1])

                                if len(ma_cs_values_list) >= 2:
                                    diff_values = difference_window(ma_cs_values_list)
                                    diff_values_list.append(diff_values[-1])

                                    if len(diff_values_list) >= window_size:
                                        ma_diff_values = moving_average(diff_values_list, window_size)
                                        ma_diff_values_list.append(ma_diff_values[-1])

                                        if len(ma_diff_values_list) >= window_size:
                                            var_values = moving_variance(ma_diff_values_list, window_size)
                                            var_values_list.append(var_values[-1])

                                            if len(var_values_list) > 1 and early_stop is False:
                                                for i in range(1, len(var_values_list)):
                                                    current_derivative = var_values_list[i] - var_values_list[i - 1]

                                                    if prev_derivative is not None:
                                                        if prev_derivative < 0 and current_derivative > 0:
                                                            valley_count += 1

                                                            if valley_count == 2:
                                                                print("find transition point")
                                                                print(f"Current Variance is: {var_values_list[-1]}")
                                                                print(f"Current Step is: {global_step_counter}")
                                                                early_stop = True
                                                                save_at_tp = True
                                                                break

                                                    prev_derivative = current_derivative
                                                valley_count = 0

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = network.apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                    if save_at_tp is True:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"save transition point")
                            weights_to_save = weights_queue[0]
                            print(f"transition point epoch:{weights_to_save['epoch']} global_step:{weights_to_save['global_step']} global_step_counter:{weights_to_save['global_step_counter']}")
                            save_at_tp = False
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, weights_to_save['global_step_counter'])
                            save_model(ckpt_name, weights_to_save['network_state'], weights_to_save['global_step_counter'], weights_to_save['epoch'])

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, weights_to_save['global_step_counter'])

                            remove_step_no = train_util.get_remove_step_no(args, weights_to_save['global_step_counter'])
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as,
                                                                                 remove_step_no)
                                remove_model(remove_ckpt_name)

                            if args.early_stop:
                                # early stop
                                print("early stop!")
                                print(f"Turning point is at step:{weights_to_save['global_step']}")
                                print(f"Well performing LoRA models are on Epoch {weights_to_save['global_step'] // train_dataset_group.num_train_images +1} and Epoch {weights_to_save['global_step'] // train_dataset_group.num_train_images +2}")
                                1 / 0

                current_loss = loss.detach().item()
                if epoch == 0:
                    loss_list.append(current_loss)
                else:
                    loss_total -= loss_list[step]
                    loss_list[step] = current_loss
                loss_total += current_loss
                avr_loss = loss_total / len(loss_list)
                logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(args, current_loss, avr_loss, lr_scheduler, keys_scaled, mean_norm, maximum_norm)
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_total / len(loss_list)}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            if args.calculate_cs:
                te_result, unet_result = compute_cosine_similarity(accelerator.device, unet, text_encoder, accelerator.unwrap_model(network).state_dict())
                cs_epoch = np.mean(unet_result)
                accelerator.print(f"Epoch [{epoch + 1}/{num_train_epochs}], unet cosine_similarity Value: {cs_epoch:.6f}")
                cs_te_epoch = np.mean(te_result)
                accelerator.print(f"Epoch [{epoch + 1}/{num_train_epochs}], tecoder cosine_similarity Value: {cs_te_epoch:.6f}")
                result = np.concatenate((te_result, unet_result), axis=1)
                cs_to_epoch = np.mean(result)
                accelerator.print(f"Epoch [{epoch + 1}/{num_train_epochs}], total cosine_similarity Value: {cs_to_epoch:.6f}")


                if cs is not None and args.logging_dir is not None:
                    writer.add_scalar('lr/epoch(param0)', optimizer.param_groups[0]['lr'], epoch + 1)
                    writer.add_scalar('lr/epoch(param1)', optimizer.param_groups[1]['lr'], epoch + 1)
                    writer.add_scalar('unet_cs/epoch', cs_epoch, epoch + 1)
                    writer.add_scalar('te_cs/epoch', cs_te_epoch, epoch + 1)
                    writer.add_scalar('total_cs/epoch', cs_to_epoch, epoch + 1)

            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as,
                                                                          remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer,
                               text_encoder, unet)


        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        writer.close()
        accelerator.end_training()

        with open(os.path.join(logging_dir, 'training_result_data.json'), 'w') as f:
            json.dump(training_cs_data, f)

        ma_results = {"cs_values_list": cs_values_list, "ma_cs_values_list": ma_cs_values_list,
                      "diff_values_list": diff_values_list, "ma_diff_values_list": ma_diff_values_list,
                      "var_values_list": var_values_list}

        with open(os.path.join(logging_dir, 'ma_results.json'), 'w') as f:
            json.dump(ma_results, f)

        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            print("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")

    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument(
        "--network_dim", type=int, default=None, help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）"
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*", help="additional argmuments for network (key=value) / ネットワークへの追加の引数"
    )
    parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument(
        "--network_train_text_encoder_only", action="store_true", help="only training Text Encoder part / Text Encoder関連部分のみ学習する"
    )
    parser.add_argument(
        "--training_comment", type=str, default=None, help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列"
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--calculate_cs",
        action="store_true",
        help="calculate cosine similarity(cs) and CS fluctuations as training progresses",
    )
    parser.add_argument(
        "--cs_interval",
        type=int,
        default=20,
        help="calculation intervals for CS and CS-Fluctuations",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="window size of moving window average",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="early stop training process base on CS-Fluctuations",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)