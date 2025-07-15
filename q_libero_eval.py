import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
# from models.seer_model import SeerAgent
from models.q_seer_model import Q_SeerAgent
from utils.q_train_utils import get_checkpoint, train_one_epoch_calvin, get_ckpt_name
from utils.q_argument_utils import get_parser
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset, get_droid_dataset, get_libero_pretrain_dataset, \
    get_libero_finetune_dataset, get_real_finetune_dataset, get_oxe_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env
from tqdm import tqdm


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@record
def main(args):
    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.save_checkpoint_path)}"
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)
    ptbs = args.world_size * args.batch_size * args.gradient_accumulation_steps
    print("training batch size:", ptbs)
    args.run_name = args.run_name.replace("Seer", f"Seer_validate_q-model")
    print("run_name:", args.run_name)
    model = Q_SeerAgent(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        value_pred_steps=args.value_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
    )
    calvin_dataset = get_libero_finetune_dataset(args, model.image_processor, clip, epoch=0)
    random_seed(args.seed, args.rank)
    print(f"Start running validation on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        print("wandb_project :", args.wandb_project)
        print("wandb_entity :", args.wandb_entity)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    elif args.precision == "fp32":
        model = model.float()
        if 'vision_encoder' in args.bf16_module:
            model.vision_encoder.bfloat16()
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()
    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    total_validation_steps = calvin_dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = calvin_dataset.dataloader.num_batches * args.warmup_epochs
    if args.rank == 0:
        print(f"Total validation steps: {total_validation_steps}")
    if args.finetune_from_pretrained_ckpt is not None:
        if args.rank == 0:
            print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_pretrained_ckpt}")
        checkpoint = torch.load(args.finetune_from_pretrained_ckpt, map_location="cpu")
        if checkpoint["model_state_dict"][
            "module.transformer_backbone_position_embedding"].shape != ddp_model.module.transformer_backbone_position_embedding.shape:
            checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"] = \
            checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"][:, :args.sequence_length,
            :, :]
        print("loading pretrained weights :", checkpoint["model_state_dict"].keys())
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)

    ddp_model.eval()
    cast_dtype = torch.float32
    calvin_loader = calvin_dataset.dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=calvin_loader.num_batches
    )
    total_cnt = success_cnt = strict_success_cnt = 0
    for num_steps, batch_calvin in t:
        # images
        images_primary = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        # text tokens
        text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)

        # states
        states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2

        input_image_primary = images_primary[:, :args.sequence_length, :]
        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]

        # permute inputs
        perm_index = torch.randperm(args.sequence_length)
        perm_input_image_primary = input_image_primary[:, perm_index]
        perm_input_image_wrist = input_image_wrist[:, perm_index]
        perm_input_text_token = input_text_token[:, perm_index]
        perm_input_state = input_state[:, perm_index]

        with torch.no_grad():
            perm_value_pred, _, _, _, _ = model(
                image_primary=perm_input_image_primary,
                image_wrist=perm_input_image_wrist,
                state=perm_input_state,
                text_token=perm_input_text_token,
            )
            perm_value_pred = perm_value_pred.squeeze(-1).squeeze(-1)

        # unpermute
        unperm_value_pred = torch.zeros_like(perm_value_pred)
        for i in range(len(perm_index)):
            unperm_value_pred[:, perm_index[i]] = perm_value_pred[:, i]

        for i in range(args.sequence_length - 1):
            for j in range(i + 1, args.sequence_length):
                value_i, value_j = unperm_value_pred[:, i], unperm_value_pred[:, j]
                delta_value = value_j - value_i
                delta_value_01 = torch.where(delta_value >= 0.0,
                                             torch.ones_like(delta_value),
                                             torch.zeros_like(delta_value))
                delta_value_01_strict = torch.where(delta_value > 0.0,
                                                    torch.ones_like(delta_value),
                                                    torch.zeros_like(delta_value))
                success_cnt += delta_value_01.sum().item()
                strict_success_cnt += delta_value_01_strict.sum().item()
                total_cnt += delta_value_01.shape[0]
        t.set_postfix({"success_rate": success_cnt / total_cnt, "strict_success_rate": strict_success_cnt / total_cnt})
    print(f'success rate = {success_cnt / total_cnt}')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
