"""Predict attribute score with off-the-shelf Place365 classifier
Adapt the code from https://github.com/genforce/higan/blob/master/synthesize.py
"""
import sys
sys.path.append("../")

import os.path
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from higan.predictors.helper import build_predictor
from higan.utils.logger import setup_logger
# from higan.utils.visualizer import HtmlPageVisualizer
# from higan.utils.visualizer import save_image
from utils import load_yaml
import dataset as dataset_module
from torch.utils.data import DataLoader


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt_root', type=str, default=os.path.join("../runs", "celeba64_baseline"))
  parser.add_argument('-N', '--num', type=int, default=0,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is valid. Otherwise '
                           'a positive number is required. (default: 0)')
  parser.add_argument('--predictor_name', type=str, default='scene',
                      help='Name of the predictor used for analysis. (default: '
                           'scene)')
  parser.add_argument('--logfile_name', type=str, default='log.txt',
                      help='Name of the log file. If not specified, log '
                           'message will be saved to path '
                           '`${OUTPUT_DIR}/log.txt` by default.')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()

  work_dir = args.ckpt_root
  work_dir += f"/../score_first{args.num}"
  logger_name = f'score_logger'
  logger = setup_logger(work_dir, args.logfile_name, logger_name)

  # prepare daloader
  config_path = os.path.join(args.ckpt_root, "config.yml")
  config = load_yaml(config_path)
  train_dataset_config = config["train_dataset_config"]
  train_dataset_config.update({
    "augmentation": False
  })
  dataset_name = train_dataset_config["name"]
  dataset = getattr(dataset_module, dataset_name, None)(train_dataset_config)
  loader = DataLoader(
    dataset=dataset,
    pin_memory=True,
    collate_fn=dataset.collate_fn,
    num_workers=4,
    batch_size=128,
    shuffle=False
  )

  # logger.info(f'Initializing generator.')
  # # model = build_generator(args.model_name, logger=logger)

  # logger.info(f'Preparing latent codes.')
  # if os.path.isfile(args.latent_codes_path):
  #   logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
  #   latent_codes = np.load(args.latent_codes_path)
  #   latent_codes = model.preprocess(latent_codes=latent_codes,
  #                                   latent_space_type=args.latent_space_type)
  # else:
  #   if args.num <= 0:
  #     raise ValueError(f'Argument `num` should be specified as a positive '
  #                      f'number since the latent code path '
  #                      f'`{args.latent_codes_path}` does not exist!')
  #   logger.info(f'  Sample latent codes randomly.')
  #   latent_codes = model.easy_sample(num=args.num,
  #                                    latent_space_type=args.latent_space_type)
  total_num = args.num if args.num < len(dataset) else len(dataset)

  # if args.generate_prediction:
  logger.info(f'Initializing predictor.')
  predictor = build_predictor(args.predictor_name)

  # if args.generate_html:
  #   viz_size = None if args.viz_size == 0 else args.viz_size
  #   visualizer = HtmlPageVisualizer(num_rows=args.html_row,
  #                                   num_cols=args.html_col,
  #                                   grid_size=total_num,
  #                                   viz_size=viz_size)

  logger.info(f'Generating {total_num} samples.')
  # results = defaultdict(list)
  predictions = defaultdict(list)
  pbar = tqdm(total=total_num, leave=False)
  # for inputs in model.get_batch_inputs(latent_codes):
  #   outputs = model.easy_synthesize(latent_codes=inputs,
  #                                   latent_space_type=args.latent_space_type,
  #                                   generate_style=args.generate_style,
  #                                   generate_image=not args.skip_image)

  num_samples_read = 0
  for batch in loader:
    # if key == 'image':
    # if args.generate_prediction:
    append_num = min(len(batch["gts"]), total_num-num_samples_read)
    if append_num <= 0:
      break
    pred_outputs = predictor.easy_predict(batch["gts"])
    for pred_key, pred_val in pred_outputs.items():
      predictions[pred_key].append(pred_val[:append_num, :])
    num_samples_read += append_num
    # for image in val:
    #   if args.save_raw_synthesis:
    #     save_image(os.path.join(work_dir, f'{pbar.n:06d}.jpg'), image)
    #   if args.generate_html:
    #     row_idx = pbar.n // visualizer.num_cols
    #     col_idx = pbar.n % visualizer.num_cols
    #     visualizer.set_cell(row_idx, col_idx, image=image)
    pbar.update(append_num)
      # else:
        # results[key].append(val)
    # if 'image' not in outputs:
    #   pbar.update(inputs.shape[0])
  # pbar.close()

  logger.info(f'Saving results.')
  # if args.generate_html:
  #   visualizer.save(os.path.join(work_dir, args.html_name))
  # for key, val in results.items():
  #   np.save(os.path.join(work_dir, f'{key}.npy'), np.concatenate(val, axis=0))
  if predictions:
    if args.predictor_name == 'scene':
      # Categories
      categories = np.concatenate(predictions['category'], axis=0)
      assert categories.shape[0] == total_num, "Did not collect enough number of categorie scores."
      detailed_categories = {
          'score': categories,
          'name_to_idx': predictor.category_name_to_idx,
          'idx_to_name': predictor.category_idx_to_name,
      }
      np.save(os.path.join(work_dir, 'category.npy'), detailed_categories)
      # Attributes
      attributes = np.concatenate(predictions['attribute'], axis=0)
      assert categories.shape[0] == total_num, "Did not collect enough number of attribute scores."
      detailed_attributes = {
          'score': attributes,
          'name_to_idx': predictor.attribute_name_to_idx,
          'idx_to_name': predictor.attribute_idx_to_name,
      }
      np.save(os.path.join(work_dir, 'attribute.npy'), detailed_attributes)
    # else:
    #   for key, val in predictions.items():
    #     np.save(os.path.join(work_dir, f'{key}.npy'),
    #             np.concatenate(val, axis=0))


if __name__ == '__main__':
  main()