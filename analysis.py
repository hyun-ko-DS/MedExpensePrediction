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


def continuous_correlation_test(df, numeric_continuous_cols, target_col):
    # 1. 수치형 - 연속형 X와 연속형 Y: Pearson 상관분석
    results = pd.DataFrame()
    for col in numeric_continuous_cols:
        # 정규성 검정 (Shapiro-Wilk test)
        _, p_value = stats.shapiro(df[col].dropna())
        is_normal = p_value > 0.05
        
        if is_normal:
            # Pearson 상관분석
            corr, p_value = stats.pearsonr(df[col].dropna(), df[target_col].dropna())
            results[col] = {
                'type': 'continuous',
                'test': 'pearson',
                'correlation': corr,
                'p_value': p_value
            }
        else:
            # 정규성 만족하지 않으면 Spearman 사용
            corr, p_value = stats.spearmanr(df[col].dropna(), df[target_col].dropna())
            results[col] = {
                'type': 'continuous',
                'test': 'spearman',
                'correlation': corr,
                'p_value': p_value
            }
    return results


def ordinal_correlation_test(df, categorical_ordinal_cols, target_col):
    """
    범주형(순서형) 변수와 연속형 Y 변수 간의 Spearman 상관분석을 수행하는 함수
    
    Parameters:
    -----------
    df : DataFrame
        분석할 데이터프레임
    categorical_ordinal_cols : list
        범주형(순서형) 변수들의 리스트
    target_col : str
        타겟 변수명
    
    Returns:
    --------
    DataFrame
        각 변수별 상관분석 결과를 담은 DataFrame
    """
    results = pd.DataFrame()
    
    for col in categorical_ordinal_cols:
        # Spearman 상관분석 수행
        corr, p_value = stats.spearmanr(df[col].dropna(), df[target_col].dropna())
        
        # 결과 저장
        results[col] = {
            'type': 'ordinal',
            'test': 'spearman',
            'correlation': corr,
            'p_value': p_value
        }
    
    return results


def nominal_anova_test(df, categorical_nominal_cols, target_col):
    """
    범주형(명목형) 변수와 연속형 Y 변수 간의 ANOVA 또는 Kruskal-Wallis 분석을 수행하는 함수
    
    Parameters:
    -----------
    df : DataFrame
        분석할 데이터프레임
    categorical_nominal_cols : list
        범주형(명목형) 변수들의 리스트
    target_col : str
        타겟 변수명
    
    Returns:
    --------
    DataFrame
        각 변수별 분석 결과를 담은 DataFrame
    """
    results = pd.DataFrame()
    
    for col in categorical_nominal_cols:
        # 각 범주별 그룹 생성
        groups = [group for _, group in df.groupby(col)[target_col]]
        
        # 데이터가 3개 미만인 그룹이 있는지 확인
        has_small_groups = any(len(group) < 3 for group in groups)
        
        if has_small_groups:
            # 작은 그룹이 있으면 바로 Kruskal-Wallis 사용
            h_stat, p_value = stats.kruskal(*groups)
            test_type = 'kruskal_wallis'
            test_stat = h_stat
            is_normal = False
            is_homogeneous = False
        else:
            # 정규성 검정 (Shapiro-Wilk test)
            normality_pvalues = [stats.shapiro(group)[1] for group in groups]
            is_normal = all(p > 0.05 for p in normality_pvalues)
            
            # 등분산성 검정 (Levene's test)
            levene_stat, levene_pvalue = stats.levene(*groups)
            is_homogeneous = levene_pvalue > 0.05
            
            if is_normal and is_homogeneous:
                # 정규성 검정 만족 시, ANOVA 분석 수행
                f_stat, p_value = stats.f_oneway(*groups)
                test_type = 'anova'
                test_stat = f_stat
            else:
                # 정규성 검정 비만족 시, Kruskal-Wallis 검정 수행
                h_stat, p_value = stats.kruskal(*groups)
                test_type = 'kruskal_wallis'
                test_stat = h_stat
        
        # 결과 저장
        results[col] = {
            'type': 'nominal',
            'test': test_type,
            'test_statistic': test_stat,
            'p_value': p_value,
            'is_normal': is_normal,
            'is_homogeneous': is_homogeneous
        }
    
    return results

# def separate_columns(df):
#   # ID 컬럼들은 제외.
#   pure_columns = list(set(df.columns.tolist()) - set(id_columns))
#   numeric_cols, categorical_cols = [], []
#   for col in pure_columns:
#       unique_values = df[col].nunique()

#       # 숫자형 데이터 & 고유값 개수가 일정 이상이면 연속형
#       if pd.api.types.is_numeric_dtype(df[col]) and unique_values >= 20:
#           numeric_cols.append(col)
#       # 그 외는 범주형
#       else:
#           categorical_cols.append(col)

#   return numeric_cols, categorical_cols


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