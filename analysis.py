import pandas as pd

# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel import compare
from linearmodels.panel import compare
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


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