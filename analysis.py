import pandas as pd

# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats


id_columns = ['HHIDWON', 'PIDWON']

def check_columns(df):
  print(f"<데이터 전체 관찰값 개수 = {len(df)}>", "\n")
  for col in df.columns:
    print(f"{col}: 관찰값 개수 = {len(df[col])}, 유니크한 관찰값 개수 = {df[col].nunique(dropna = False)}, 결측치 비율 = {100 * round(df[col].isnull().sum() / len(df[col]), 4)}")

def separate_columns(df):
  # ID 컬럼들은 제외.
  pure_columns = list(set(df.columns.tolist()) - set(id_columns))
  numeric_cols, categorical_cols = [], []
  for col in pure_columns:
      unique_values = df[col].nunique()

      # 숫자형 데이터 & 고유값 개수가 일정 이상이면 연속형
      if pd.api.types.is_numeric_dtype(df[col]) and unique_values >= 20:
          numeric_cols.append(col)
      # 그 외는 범주형
      else:
          categorical_cols.append(col)

  return numeric_cols, categorical_cols


# def continuous_correlation_test(df, continuous_cols, target_col):
#        # 1. 연속형 X와 연속형 Y: Pearson 상관분석
#     for col in continuous_cols:
#         # 정규성 검정
#         _, p_value = stats.normaltest(df[col].dropna())
#         is_normal = p_value > 0.05
        
#         if is_normal:
#             # Pearson 상관분석
#             corr, p_value = stats.pearsonr(df[col].dropna(), df[target_col].dropna())
#             results[col] = {
#                 'type': 'continuous',
#                 'test': 'pearson',
#                 'correlation': corr,
#                 'p_value': p_value
#             }
#         else:
#             # 정규성 만족하지 않으면 Spearman 사용
#             corr, p_value = stats.spearmanr(df[col].dropna(), df[target_col].dropna())
#             results[col] = {
#                 'type': 'continuous',
#                 'test': 'spearman',
#                 'correlation': corr,
#                 'p_value': p_value
#             }

# def ordinal_correlation_test(df, ordinal_cols, target_col):
#     # 2. 순서형 X와 연속형 Y: Spearman 상관분석
#   for col in ordinal_cols:
#       corr, p_value = stats.spearmanr(df[col].dropna(), df[target_col].dropna())
#       results[col] = {
#           'type': 'ordinal',
#           'test': 'spearman',
#           'correlation': corr,
#           'p_value': p_value
#       }

#   return pass
    
#     # 3. 명목형 X와 연속형 Y: ANOVA
#     for col in nominal_cols:
#         # 각 범주별 평균 계산
#         groups = [group for _, group in df.groupby(col)[target_col]]
#         f_stat, p_value = stats.f_oneway(*groups)
#         results[col] = {
#             'type': 'nominal',
#             'test': 'anova',
#             'f_statistic': f_stat,
#             'p_value': p_value
#         }
    

# def analyze_correlations(df: pd.DataFrame, target_col: str) -> dict:
#     """
#     변수들 간의 상관관계를 분석합니다.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         입력 데이터프레임
#     target_col : str
#         타겟 변수명 (Y 변수)
    
#     Returns:
#     --------
#     dict
#         각 변수별 상관분석 결과
#     """
#     # ID 컬럼 제외
#     pure_columns = list(set(df.columns.tolist()) - set(id_columns + [target_col]))
    
#     # 연속형 변수 추출
#     continuous_cols = []
#     for col in pure_columns:
#         if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() >= 20:
#             continuous_cols.append(col)
    
#     # 범주형 변수 구분
#     ordinal_cols, nominal_cols = separate_categorical_columns(df)
    
#     results = {}
    

    

    
#     return results