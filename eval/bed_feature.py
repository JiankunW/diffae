import sys
sys.path.append("../")
import os.path
import argparse

import numpy as np
from tqdm import tqdm
import torch

from utils import load_yaml
import eval.dataset as dataset_module
from torch.utils.data import DataLoader
# import model.representation.encoder as encoder_module
from experiment import LitModel
from templates import bedroom128_autoenc

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt_root', type=str, default=os.path.join("../runs", "bedroom128_official"))
  parser.add_argument('--ckpt_name', type=str, default="last", help="name without .ckpt")
  parser.add_argument('--gpu', type=str, default="0")
  parser.add_argument('--no-ema', action="store_false", help="load ema weights for the encoder")
  parser.add_argument('-N', '--num', type=int, default=0,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is valid. Otherwise '
                           'a positive number is required. (default: 0)')
  return parser.parse_args()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main():
  """Main function."""
  args = parse_args()

  work_dir = args.ckpt_root
  work_dir += f"/features_{args.ckpt_name}_first{args.num}"
  if args.no_ema:
    work_dir += "_ema"
  os.makedirs(work_dir, exist_ok=True)
  # prepare daloader
  conf = bedroom128_autoenc()
  # config = AttrDict(config)
  # config.train_mode = TrainMode.latent_diffusion
  # import pdb; pdb.set_trace()
  dataset_config = {
      "name": "BEDROOM",
      "data_path": "../datasets/bedroom256.lmdb",
      "image_size": 128,
      "image_channel": 3,
      "augmentation": False,
  }
  dataset_name = dataset_config["name"]
  dataset = getattr(dataset_module, dataset_name, None)(dataset_config)
  loader = DataLoader(
    dataset=dataset,
    pin_memory=True,
    collate_fn=dataset.collate_fn,
    num_workers=4,
    batch_size=128,
    shuffle=False
  )
  total_num = args.num if args.num < len(dataset) else len(dataset)

  #########################

  device = f"cuda:{args.gpu}"
  ckpt_path = os.path.join(args.ckpt_root, "checkpoints", f"{args.ckpt_name}.ckpt")
  state = torch.load(ckpt_path, map_location=torch.device('cpu'))
  model = LitModel(conf)
  model.load_state_dict(state['state_dict'])
  encoder = model.ema_model.encoder if args.no_ema else model.model.encoder
  import pdb; pdb.set_trace()
  # encoder = getattr(encoder_module, config["encoder_config"]["model"], None)(**config["encoder_config"])
  # encoder.load_state_dict(ckpt['ema_encoder'] if args.no_ema else ckpt['encoder'])
  encoder.to(device).eval()

  # 1. infer feat stats
  # define file name to store/load stats
  stats_name = f"stats_{args.ckpt_name}"
  if args.no_ema:
    stats_name += "_ema"
  stats_name += ".pt"

  z_list = []
  if os.path.exists(os.path.join(args.ckpt_root, stats_name)):
      data = torch.load(os.path.join(args.ckpt_root, stats_name))
      mean = data["mean"]
      std = data["std"]
  else:
      with torch.inference_mode():
          for batch in tqdm(loader, desc="infer feature stats"):
              x_0 = batch["net_input"]["x_0"].to(device, non_blocking=True)
              z = encoder(x_0)
              z_list.append(z.cpu())

          latent = torch.cat(z_list,dim=0)
          mean = latent.mean(0)
          std = latent.std(0)
          torch.save({"mean": mean, "std":std}, os.path.join(args.ckpt_root, stats_name))
  mean = mean.to(device)
  std = std.to(device)

  # 2. infer features
  f_list = []
  pbar = tqdm(total=total_num, desc="infer normalized features")
  num_samples_read = 0
  for batch in loader:
    # if key == 'image':
    # if args.generate_prediction:
    append_num = min(len(batch["gts"]), total_num-num_samples_read)
    if append_num <= 0:
      break
    imgs = batch["net_input"]["x_0"].to(device, non_blocking=True)
    with torch.inference_mode():
        feat = encoder(imgs)
        feat = (feat - mean) / std
    f_list.append(feat.cpu().numpy()[:append_num, :])
    num_samples_read += append_num
    pbar.update(append_num)

  # 3. save results
  fs = np.concatenate(f_list, axis=0)
  assert fs.shape[0] == total_num, "Did not collect enough number of features."
  np.save(os.path.join(work_dir, 'feats.npy'), fs)

if __name__ == '__main__':
  main()