import pandas as pd
import numpy as np

# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso    
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects, compare
from statsmodels.tools.tools import add_constant

from preprocessing import move_year_column, move_y_column



id_columns = ['HHIDWON', 'PIDWON']

def check_columns(df):
  print(f"<데이터 전체 관찰값 개수 = {len(df)}>", "\n")
  for col in df.columns:
    print(f"{col}: 관찰값 개수 = {len(df[col])}, 유니크한 관찰값 개수 = {df[col].nunique(dropna = False)}, 결측치 비율 = {100 * round(df[col].isnull().sum() / len(df[col]), 4)}")


def train_test_split_by_year(df: pd.DataFrame, target_col: str = 'medical_expense') -> tuple:
    """
    연도 기반으로 train-test split을 수행하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    target_col : str
        타겟 변수명 (기본값: 'medical_expense')
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # train 데이터 (2014-2017)
    train_data = move_year_column(df[df['YEAR'].isin([2014, 2015, 2016, 2017])].copy())
    
    # test 데이터 (2018)
    test_data = move_year_column(df[df['YEAR'] == 2018].copy()) 

    print(f"Train data years: {train_data['YEAR'].unique()}")
    print(f"Test data years: {test_data['YEAR'].unique()}")

    return train_data, test_data

def X_y_split_by_year(train: pd.DataFrame, test: pd.DataFrame, target_col: str = 'medical_expense', is_panel: bool = True) -> tuple:    
    # 타겟 변수 분리
    y_train = train[target_col].reset_index(drop=True)
    y_test = test[target_col].reset_index(drop=True)

    # 특성 변수 분리 (타겟 변수와 YEAR 제외)
    X_train = train.drop([target_col], axis=1).reset_index(drop=True)
    X_test = test.drop([target_col], axis=1).reset_index(drop=True)

    # 데이터 크기 출력
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    if not is_panel:
        X_train = X_train.drop(['HHIDWON', 'PIDWON', 'YEAR'], axis=1)
        X_test = X_test.drop(['HHIDWON', 'PIDWON', 'YEAR'], axis=1)

    return X_train, X_test, y_train, y_test


def check_y_by_year(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby('YEAR').agg(
    num_observations = ('medical_expense', 'count'),
    median_med_exp = ('medical_expense', 'median'),
    mean_med_exp = ('medical_expense', 'mean'),
    std_med_exp = ('medical_expense', np.std)
    )
    return result


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


def analyze_categorical_continuous_relationship(df, categorical_cols, target_col):
    """
    범주형 변수와 연속형 타겟 변수 간의 관계를 분석합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    categorical_cols : list
        범주형 변수 컬럼 리스트
    target_col : str
        타겟 변수 컬럼명
    
    Returns:
    --------
    pandas.DataFrame
        분석 결과
    """
    results = {}
    
    for col in categorical_cols:
        # 각 범주별 그룹 생성
        groups = [group for _, group in df.groupby(col)[target_col] if len(group) >= 3]
        unique_values = df[col].nunique()
        
        if unique_values < 2:
            # 단일 그룹인 경우
            results[col] = {
                'type': 'nominal',
                'test': 'skipped',
                'reason': 'single_group',
                'correlation': np.nan,
                'effect_size': np.nan,
                'p_value': np.nan
            }
        elif unique_values == 2:
            # 이분형 변수인 경우 Point-Biserial Correlation
            try:
                # 0과 1로 변환
                binary_values = pd.get_dummies(df[col], drop_first=True).iloc[:, 0]
                correlation, p_value = stats.pointbiserialr(binary_values, df[target_col])
                
                # Cohen's d 계산
                group1 = df[df[col] == df[col].unique()[0]][target_col]
                group2 = df[df[col] == df[col].unique()[1]][target_col]
                cohens_d = (group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2)
                
                results[col] = {
                    'type': 'nominal',
                    'test': 'point_biserial',
                    'correlation': correlation,
                    'effect_size': cohens_d,
                    'p_value': p_value
                }
            except:
                results[col] = {
                    'type': 'nominal',
                    'test': 'failed',
                    'reason': 'calculation_error',
                    'correlation': np.nan,
                    'effect_size': np.nan,
                    'p_value': np.nan
                }
        else:
            # 3개 이상의 그룹이 있는 경우 ANOVA/Kruskal-Wallis
            if len(groups) < 2:
                results[col] = {
                    'type': 'nominal',
                    'test': 'skipped',
                    'reason': 'insufficient_groups',
                    'correlation': np.nan,
                    'effect_size': np.nan,
                    'p_value': np.nan
                }
                continue
                
            try:
                # 정규성 검정
                normality_pvalues = [stats.shapiro(group)[1] for group in groups]
                is_normal = all(p > 0.05 for p in normality_pvalues)
                
                # 등분산성 검정
                levene_stat, levene_pvalue = stats.levene(*groups)
                is_homogeneous = levene_pvalue > 0.05
                
                if is_normal and is_homogeneous:
                    # ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    df_residual = sum(len(group) for group in groups) - len(groups)
                    eta_squared = f_stat / (f_stat + df_residual)
                    
                    results[col] = {
                        'type': 'nominal',
                        'test': 'anova',
                        'correlation': np.sqrt(eta_squared),  # eta-squared의 제곱근을 상관계수로 사용
                        'effect_size': eta_squared,
                        'p_value': p_value
                    }
                else:
                    # Kruskal-Wallis
                    h_stat, p_value = stats.kruskal(*groups)
                    df_residual = sum(len(group) for group in groups) - len(groups)
                    eta_squared = h_stat / (h_stat + df_residual)
                    
                    results[col] = {
                        'type': 'nominal',
                        'test': 'kruskal',
                        'correlation': np.sqrt(eta_squared),
                        'effect_size': eta_squared,
                        'p_value': p_value
                    }
            except:
                results[col] = {
                    'type': 'nominal',
                    'test': 'failed',
                    'reason': 'calculation_error',
                    'correlation': np.nan,
                    'effect_size': np.nan,
                    'p_value': np.nan
                }
    
    return pd.DataFrame(results).T