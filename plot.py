import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

id_columns = ['HHIDWON', 'PIDWON']

# 변수 시각화
def draw_histogram(df, continuous_columns):
  # 컬럼 이름 순서대로 정렬
  continuous_columns = sorted(continuous_columns)
  total_plots = len(continuous_columns)
  cols = 2
  rows = math.ceil(total_plots / cols)
  plt.figure(figsize = (4 * cols, 3 * rows))

  for i, col in enumerate(continuous_columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(x = df[col].dropna(), bins = 30, kde = True, color=sns.color_palette('pastel')[1])
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
  plt.show()

# 시각화 함수 (이산형 변수 countplot)
def draw_countplot(df, categorical_columns):
    # 컬럼 이름 순서대로 정렬
    categorical_columns = sorted(categorical_columns)
    total_plots = len(categorical_columns)
    cols = 2
    rows = math.ceil(total_plots / cols)
    plt.figure(figsize=(4 * cols, 3 * rows))

    for i, col in enumerate(categorical_columns, 1):
      plt.subplot(rows, cols, i)

      # NaN 값을 포함한 경우, NaN을 제거한 후 카운트 정렬
      valid_values = df[col].dropna().astype(str)  # NaN 제거 후 문자열로 변환
      order_values = sorted(valid_values.value_counts().index)  # 알파벳 순으로 정렬

      sns.countplot(x=valid_values, order=order_values, color=sns.color_palette('pastel')[1])
      plt.xticks(rotation=45)
      plt.title(f"Countplot of {col}")
      plt.tight_layout()

    plt.show()

def draw_scatterplot(df, continuous_columns):
  # 컬럼 이름 순서대로 정렬
  continuous_columns = sorted(continuous_columns)
  total_plot = len(continuous_columns)
  cols = 3
  rows = math.ceil(total_plot / cols)
  plt.figure(figsize = (4 * cols, 3 * rows))

  for i, col in enumerate(continuous_columns, 1):
    plt.subplot(rows, cols, i)
    sns.regplot(x=col, y='medical_expense', scatter=True, line_kws={"color": "red"}, data = df, color=sns.color_palette('pastel')[1])
    plt.tight_layout()
  plt.show()

def draw_boxplot(df, columns):
  # 컬럼 이름 순서대로 정렬
  columns = sorted(columns)
  total_plot = len(columns)
  cols = 3
  rows = math.ceil(total_plot / cols)
  plt.figure(figsize = (4 * cols, 3 * rows))

  for i, col in enumerate(columns, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x = col, y = 'medical_expense', data = df, color=sns.color_palette('pastel')[1])
    plt.tight_layout()
  plt.show()