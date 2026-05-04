"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
(4) Save generated data to CSV with date column
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def generate_dates(seq_len, date_format, freq='D'):
  """Generate a sequence of dates for a synthetic sample.

  Random start year between 2000-2012.

  Args:
    - seq_len: length of the date sequence
    - date_format: output date format string
    - freq: pandas frequency string (default 'D' for daily)

  Returns:
    - dates: list of formatted date strings
  """
  year = np.random.randint(2000, 2013)
  month = np.random.randint(1, 13)
  day = np.random.randint(1, 29)
  start_date = pd.Timestamp(year=year, month=month, day=day)
  dates = pd.date_range(start=start_date, periods=seq_len, freq=freq)
  return dates.strftime(date_format).tolist()


def main (args):
  """Main function for timeGAN experiments.

  Args:
    - data_path: path to CSV file, or 'sine' for built-in sine data
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation

  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  info_dict = None
  if args.data_path == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
  else:
    ori_data, info_dict = real_data_loading(args.data_path, args.seq_len)

  print(args.data_path + ' dataset is ready.')

  ## Synthetic data generation by TimeGAN
  # Set network parameters
  parameters = dict()
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size

  generated_data = timegan(ori_data, parameters)
  print('Finish Synthetic Data Generation')

  ## Performance metrics
  # Output initialization
  metric_results = dict()

  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)

  metric_results['discriminative'] = np.mean(discriminative_score)

  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)

  metric_results['predictive'] = np.mean(predictive_score)

  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca')
  visualization(ori_data, generated_data, 'tsne')

  ## Print discriminative and predictive scores
  print(metric_results)

  ## Save generated data to CSV
  if info_dict is not None:
    # Reverse the data_loading normalization to restore original scale
    data_min = info_dict['data_min']
    data_max = info_dict['data_max']
    save_data = [g * (data_max - data_min + 1e-7) + data_min for g in generated_data]

    feature_cols = info_dict['feature_columns']
    original_columns = info_dict['original_columns']

    all_rows = []
    for sample in save_data:
      sample_df = pd.DataFrame(sample, columns=feature_cols)
      # Add date column if original had one
      if info_dict.get('has_date', False):
        date_col = info_dict['date_column']
        date_format = info_dict['date_format']
        date_freq = info_dict['date_freq']
        sample_df[date_col] = generate_dates(len(sample), date_format, date_freq)
        # Reorder columns to match original CSV
        sample_df = sample_df[original_columns]
      all_rows.append(sample_df)

    final_df = pd.concat(all_rows, ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    final_df.to_csv(out_path, index=False)
    print('Saved generated data to ' + out_path)

  return ori_data, generated_data, metric_results


if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_path',
      help='path to CSV file (or sine for built-in sine data)',
      default='data/stock_data.csv',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--output_dir',
      help='output directory for generated CSV',
      default='generated',
      type=str)
  parser.add_argument(
      '--output_name',
      help='output filename for generated CSV',
      default='generated_data.csv',
      type=str)

  args = parser.parse_args()

  # Calls main function
  ori_data, generated_data, metrics = main(args)
