import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

id_columns = ['HHIDWON', 'PIDWON']

# 변수 시각화
def draw_histogram(df, continuous_columns):
  # 컬럼 이름 순서대로 정렬
  continuous_columns = sorted(continuous_columns)
  total_plots = len(continuous_columns)
  cols = 4
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
    cols = 4
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
  cols = 4
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
  cols = 4
  rows = math.ceil(total_plot / cols)
  plt.figure(figsize = (4 * cols, 3 * rows))

  for i, col in enumerate(columns, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x = col, y = 'medical_expense', data = df, color=sns.color_palette('pastel')[1])
    plt.tight_layout()
  plt.show()


def draw_kde_plot(df, numeric_continuous_cols):
    plt.figure(figsize=(12, 6))
    for col in numeric_continuous_cols:
        # 데이터 정규화
        normalized_data = (df[col] - df[col].mean()) / df[col].std()
        sns.kdeplot(data=normalized_data, label=col, alpha=0.5)

    plt.title('Numeric Continuous Variables Distribution (Normalized)', pad=15)
    plt.xlabel('Standardized Values')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # x축 범위 설정
    plt.xlim(-3, 3)  # 표준정규분포의 주요 범위

    plt.tight_layout()
    plt.show()


def visualize_categorical_relationships(results_df):
    # skipped된 변수 제외
    valid_results = results_df[results_df['test'] != 'skipped']
    
    # 변수 개수에 따라 heatmap 개수 결정
    total_vars = len(valid_results)
    vars_per_plot = 10  # 각 heatmap당 10개 변수
    num_plots = math.ceil(total_vars / vars_per_plot)
    
    # 각 heatmap에 medical_expense 컬럼 추가
    for i in range(num_plots):
        start_idx = i * vars_per_plot
        end_idx = min((i + 1) * vars_per_plot, total_vars)
        
        # 현재 plot에 해당하는 변수들 선택
        current_vars = valid_results.iloc[start_idx:end_idx]
        
        # 상관계수만 추출하여 데이터프레임 생성 (float 타입으로 명시적 변환)
        correlation_df = pd.DataFrame({
            'medical_expense': current_vars['correlation'].astype(float)
        })
        
        # 상삼각 마스크 생성
        mask = np.triu(np.ones_like(correlation_df, dtype=bool))
        
        # 시각화
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_df, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .5})
        
        plt.title(f'Correlation Strength with Medical Expense (Plot {i+1}/{num_plots})', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    # 전체 결과 출력
    print("\n상관관계 분석 결과:")
    print(valid_results[['test', 'correlation', 'effect_size', 'p_value']].sort_values('correlation', ascending=False))
    
    return valid_results.sort_values('correlation', ascending=False)

