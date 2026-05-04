"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - Supports arbitrary CSV files with optional date column
  - Non-overlapping sequence extraction (step size = seq_len)
"""

## Necessary Packages
import numpy as np
import pandas as pd


def MinMaxScaler(data):
  """Min Max normalizer.

  Args:
    - data: original data

  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def infer_date_format(date_series):
  """Infer date format string from a pandas Series.

  Args:
    - date_series: pandas Series containing date strings

  Returns:
    - fmt: inferred date format string
  """
  formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y',
             '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
             '%d-%m-%Y %H:%M:%S', '%Y%m%d', '%d.%m.%Y', '%Y.%m.%d',
             '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%dT%H:%M:%S',
             '%Y-%m-%d %I:%M:%S %p', '%m/%d/%Y %I:%M:%S %p']
  best_fmt = '%Y-%m-%d'
  best_count = 0
  for fmt in formats:
    try:
      parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
      count = parsed.notna().sum()
      if count > best_count:
        best_count = count
        best_fmt = fmt
    except:
      continue
  return best_fmt


def detect_date_column(df):
  """Detect date column in DataFrame.

  Args:
    - df: pandas DataFrame

  Returns:
    - date_col: name of date column or None
    - date_example: first raw date string as example or None
    - freq: inferred date frequency or 'D'
  """
  date_keywords = ['date', 'time', 'timestamp', 'datetime']

  # First check column names containing date-related keywords
  for col in df.columns:
    if any(kw in col.lower() for kw in date_keywords):
      try:
        parsed = pd.to_datetime(df[col], errors='coerce')
        if parsed.notna().sum() > len(df) * 0.5:
          example = str(df[col].dropna().iloc[0])
          freq = infer_freq(df[col])
          return col, example, freq
      except:
        continue

  # Then check if any object/datetime column can be parsed as date
  for col in df.columns:
    if df[col].dtype == 'object' or 'datetime' in str(df[col].dtype):
      try:
        parsed = pd.to_datetime(df[col], errors='coerce')
        if parsed.notna().sum() > len(df) * 0.5:
          example = str(df[col].dropna().iloc[0])
          freq = infer_freq(df[col])
          return col, example, freq
      except:
        continue
  return None, None, 'D'


def infer_freq(date_series):
  """Infer date frequency from a pandas Series.

  Args:
    - date_series: pandas Series containing dates

  Returns:
    - freq: inferred frequency string
  """
  try:
    dates = pd.to_datetime(date_series).dropna().sort_values()
    if len(dates) < 2:
      return 'D'
    freq = pd.infer_freq(dates)
    if freq is not None:
      return freq
    # Manual fallback: compute median diff in seconds
    diffs = dates.diff().dropna()
    median_seconds = diffs.median().total_seconds()
    if median_seconds <= 60:
      return 'T'
    elif median_seconds <= 3600:
      return 'H'
    elif median_seconds <= 7200:
      return '2H'
    elif median_seconds <= 86400:
      return 'D'
    elif median_seconds <= 604800:
      return 'W'
    else:
      return 'D'
  except:
    pass
  return 'D'


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.

  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

  Returns:
    - data: generated data
  """
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)
      phase = np.random.uniform(0, 0.1)

      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
      temp.append(temp_data)

    # Align row/column
    temp = np.transpose(np.asarray(temp))
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)

  return data


def real_data_loading (data_path, seq_len):
  """Load and preprocess real-world datasets from CSV.

  Args:
    - data_path: path to CSV file
    - seq_len: sequence length

  Returns:
    - data: preprocessed data (list of np arrays)
    - info_dict: dictionary containing metadata for reconstruction
  """
  # Read CSV with pandas
  df = pd.read_csv(data_path)

  # Detect date column
  date_col, date_example, date_freq = detect_date_column(df)
  has_date = date_col is not None

  # Keep only numeric columns as features
  numeric_df = df.select_dtypes(include=[np.number])

  # Preserve original column order (date + numeric columns)
  if has_date:
    original_columns = [c for c in df.columns if c == date_col or c in numeric_df.columns]
  else:
    original_columns = numeric_df.columns.tolist()

  # Convert features to numpy array
  ori_data = numeric_df.values.astype(np.float32)

  # Flip the data to make chronological data
  ori_data = ori_data[::-1]

  # Normalize the data
  data_min = np.min(ori_data, axis=0)
  data_max = np.max(ori_data, axis=0)
  ori_data = MinMaxScaler(ori_data)

  # Preprocess the dataset (non-overlapping)
  temp_data = []
  # Cut data by sequence length without overlap
  for i in range(0, len(ori_data) - seq_len + 1, seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)

  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])

  # Build info_dict for post-processing
  info_dict = {
    'has_date': has_date,
    'date_column': date_col,
    'date_example': date_example,
    'date_freq': date_freq,
    'feature_columns': numeric_df.columns.tolist(),
    'original_columns': original_columns,
    'data_min': data_min,
    'data_max': data_max,
  }

  return data, info_dict
