# Value model based on Seer-libero
## Installation
**(1) Conda Env**
```
conda create -n seer python=3.10
conda activate seer
```

**(2) LIBERO Env**
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install transformers==4.40.2
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Running
### Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [website](https://drive.google.com/drive/folders/1zwqGvKKtjyuWdDaNSLVGJprJMPoSqAPk?usp=drive_link).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **save_checkpoint_path** to the parent directory where your experiment checkpoints are saved.  Recommend to create a ```checkpoints``` folder in the project root directory.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.```(You can find this in /data2/shujunyang/Seer-libero/checkpoints).```
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_checkpoint_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)). Recommend to be stored in ```checkpoints/vit_mae/mae_pretrain_vit_base.pth```.
    * **libero_path** to the location of LIBERO dir.

### Fine-tune the VLA to value model

```bash
# Fine-tune seer-libero to q-seer-libero on LIBERO-10 dataset
bash scripts/LIBERO_LONG/Seer/q_finetune.sh
```

```bash
# Eval q-seer-libero
bash scripts/LIBERO_LONG/Seer/q_eval.sh
```

