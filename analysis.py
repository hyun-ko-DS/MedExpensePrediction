import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso    
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import chi2_contingency, norm, pointbiserialr, kruskal, f_oneway, levene, shapiro, pearsonr, spearmanr
from linearmodels.panel import PanelOLS, RandomEffects, compare 
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor  

from preprocessing import move_year_column, move_y_column
    
id_columns = ['HHIDWON', 'PIDWON']
plot_path = 'plots'

#############################################
# 1. 데이터 검사 및 기본 정보 확인 함수들
#############################################

def check_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 각 컬럼에 대한 기본 정보를 출력하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    
    Returns:
    --------
    None
        각 컬럼별 관찰값 개수, 유니크한 값 개수, 결측치 비율을 출력
    """
    print(f"<데이터 전체 관찰값 개수 = {len(df)}>", "\n")
    for col in df.columns:
        print(f"{col}: 관찰값 개수 = {len(df[col])}, 유니크한 관찰값 개수 = {df[col].nunique(dropna = False)}, 결측치 비율 = {100 * round(df[col].isnull().sum() / len(df[col]), 4)}")

def check_y_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    연도별 타겟 변수의 통계량을 계산하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        연도별 관찰값 개수, 중앙값, 평균, 표준편차
    """
    result = df.groupby('YEAR').agg(
        num_observations = ('medical_expense', 'count'),
        median_med_exp = ('medical_expense', 'median'),
        mean_med_exp = ('medical_expense', 'mean'),
        std_med_exp = ('medical_expense', np.std)
    )
    return result

#############################################
# 2. 데이터 분할 및 전처리 함수들
#############################################

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
        (train_data, test_data)
        - train_data: 2014-2017년 데이터
        - test_data: 2018년 데이터
    """
    # train 데이터 (2014-2017)
    train_data = move_year_column(df[df['YEAR'].isin([2014, 2015, 2016, 2017])].copy())
    
    # test 데이터 (2018)
    test_data = move_year_column(df[df['YEAR'] == 2018].copy()) 

    print(f"Train data years: {train_data['YEAR'].unique()}")
    print(f"Test data years: {test_data['YEAR'].unique()}")

    return train_data, test_data

def X_y_split_by_year(train: pd.DataFrame, test: pd.DataFrame, target_col: str = 'medical_expense', is_panel: bool = True) -> tuple:    
    """
    train과 test 데이터를 X(특성)와 y(타겟)로 분리하는 함수
    
    Parameters:
    -----------
    train : pandas.DataFrame
        학습 데이터
    test : pandas.DataFrame
        테스트 데이터
    target_col : str
        타겟 변수명
    is_panel : bool
        패널 데이터 여부 (기본값: True)
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
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

#############################################
# 3. 상관관계 분석 함수들
#############################################

def continuous_correlation_test(df: pd.DataFrame, numeric_continuous_cols: list, target_col: str) -> pd.DataFrame:
    """
    연속형 변수와 타겟 변수 간의 상관관계를 분석하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    numeric_continuous_cols : list
        연속형 변수 컬럼 리스트
    target_col : str
        타겟 변수명
    
    Returns:
    --------
    pandas.DataFrame
        각 변수별 상관분석 결과 (Pearson 또는 Spearman)
    """
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

def ordinal_correlation_test(df: pd.DataFrame, categorical_ordinal_cols: list, target_col: str) -> pd.DataFrame:
    """
    순서형 변수와 타겟 변수 간의 Spearman 상관분석을 수행하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    categorical_ordinal_cols : list
        순서형 변수 컬럼 리스트
    target_col : str
        타겟 변수명
    
    Returns:
    --------
    pandas.DataFrame
        각 변수별 Spearman 상관분석 결과
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

def analyze_categorical_continuous_relationship(df: pd.DataFrame, categorical_cols: list, target_col: str) -> pd.DataFrame:
    """
    범주형 변수와 연속형 타겟 변수 간의 관계를 분석하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    categorical_cols : list
        범주형 변수 컬럼 리스트
    target_col : str
        타겟 변수명
    
    Returns:
    --------
    pandas.DataFrame
        각 변수별 분석 결과 (Point-Biserial, ANOVA, Kruskal-Wallis)
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
                        'correlation': np.sqrt(eta_squared),
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
    
    return pd.DataFrame(results)

#############################################
# 4. 이상치 처리 함수들
#############################################

def remove_outliers_iqr(df: pd.DataFrame, column: str = 'medical_expense', lower_bound: float = 0.25, 
                        upper_bound: float = 0.75, multiplier: float = 1.5) -> tuple:
    """
    IQR 방법을 사용하여 이상치를 제거하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    column : str
        이상치를 제거할 컬럼명
    lower_bound : float
        하위 사분위수 경계
    upper_bound : float
        상위 사분위수 경계
    multiplier : float
        IQR 곱셈 계수
    
    Returns:
    --------
    tuple
        (이상치가 제거된 데이터프레임, 제거된 이상치 개수, 이상치 비율)
    """
    # 원본 데이터의 행 수 저장
    original_len = len(df)
    
    # 사분위수 계산
    Q1 = df[column].quantile(lower_bound)
    Q3 = df[column].quantile(upper_bound)
    IQR = Q3 - Q1
    
    # 이상치 경계 계산
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # 이상치가 아닌 데이터만 선택
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # 제거된 이상치의 개수와 비율 계산
    removed_count = original_len - len(df_cleaned)
    removed_ratio = removed_count / original_len
    
    # 결과 출력
    print(f"\n{column} 컬럼의 이상치 제거 결과:")
    print(f"Q1 (25%): {Q1:,.2f}")
    print(f"Q3 (75%): {Q3:,.2f}")
    print(f"IQR: {IQR:,.2f}")
    print(f"하위 경계: {lower_bound:,.2f}")
    print(f"상위 경계: {upper_bound:,.2f}")
    print(f"제거된 이상치 개수: {removed_count:,}개")
    print(f"제거된 이상치 비율: {removed_ratio:.2%}")
    
    # 이상치 제거 전후 분포 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 데이터 분포
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=column, bins=50)
    plt.axvline(x=lower_bound, color='r', linestyle='--', label='Lower Bound')
    plt.axvline(x=upper_bound, color='r', linestyle='--', label='Upper Bound')
    plt.title(f'Original {column} Distribution')
    plt.legend()
    
    # 이상치 제거 후 분포
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_cleaned, x=column, bins=50)
    plt.title(f'Cleaned {column} Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # 박스플롯으로 비교
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, y=column)
    plt.title(f'Original {column} Boxplot')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_cleaned, y=column)
    plt.title(f'Cleaned {column} Boxplot')
    
    plt.tight_layout()
    plt.show()
    
    return df_cleaned, removed_count, removed_ratio

def winsorize_data(df: pd.DataFrame, column: str='medical_expense', lower_percentile: int=5, upper_percentile: int=95) -> tuple:
    """
    Winsorization을 수행하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    column : str
        Winsorization을 수행할 컬럼명
    lower_percentile : float
        하위 경계 백분위수
    upper_percentile : float
        상위 경계 백분위수
    
    Returns:
    --------
    tuple
        (Winsorization된 데이터프레임, 대체된 값의 개수, 대체된 값의 비율)
    """
    # 원본 데이터프레임 복사
    df_winsorized = df.copy()
    
    # 경계값 계산
    lower_bound = df[column].quantile(lower_percentile / 100)
    upper_bound = df[column].quantile(upper_percentile / 100)
    
    # 대체 전 값 저장
    original_values = df[column].copy()
    
    # Winsorization 수행
    df_winsorized[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    # 대체된 값의 개수와 비율 계산
    replaced_count = ((df_winsorized[column] != original_values) & 
                     (original_values.notna())).sum()
    replaced_ratio = replaced_count / len(df)
    
    # 결과 출력
    print(f"\n{column} 컬럼의 Winsorization 결과:")
    print(f"하위 경계 ({lower_percentile}%): {lower_bound:,.2f}")
    print(f"상위 경계 ({upper_percentile}%): {upper_bound:,.2f}")
    print(f"대체된 값의 개수: {replaced_count:,}개")
    print(f"대체된 값의 비율: {replaced_ratio:.2%}")
    
    # Winsorization 전후 분포 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 데이터 분포
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=column, bins=50)
    plt.axvline(x=lower_bound, color='r', linestyle='--', label='Lower Bound')
    plt.axvline(x=upper_bound, color='r', linestyle='--', label='Upper Bound')
    plt.title(f'Original {column} Distribution')
    plt.legend()
    
    # Winsorization 후 분포
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_winsorized, x=column, bins=50)
    plt.title(f'Winsorized {column} Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # 박스플롯으로 비교
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, y=column)
    plt.title(f'Original {column} Boxplot')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_winsorized, y=column)
    plt.title(f'Winsorized {column} Boxplot')
    
    plt.tight_layout()
    plt.show()
    
    # 대체된 값들의 분포 확인
    replaced_values = df[column][df_winsorized[column] != original_values]
    if len(replaced_values) > 0:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=replaced_values, bins=30)
        plt.title('Distribution of Replaced Values')
        plt.show()
    
    return df_winsorized, replaced_count, replaced_ratio

#############################################
# 5. 시계열 특성 생성 함수들
#############################################

def create_lag_columns(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """
    지정된 컬럼들의 lag 컬럼을 생성하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    target_cols : list
        lag 컬럼을 생성할 대상 컬럼들의 리스트
    
    Returns:
    --------
    pandas.DataFrame
        lag 컬럼이 추가된 데이터프레임
    """
    # 데이터프레임 복사
    result_df = df.copy()
    
    # 연도별로 정렬
    result_df = result_df.sort_values(['PIDWON', 'YEAR'])
    
    # 각 대상 컬럼에 대해 1~4년 전의 lag 컬럼 생성
    for col in target_cols:
        for lag in range(1, 5):
            result_df[f'{col}_lag_{lag}'] = result_df.groupby('PIDWON')[col].shift(lag)
    
    # lag 컬럼의 결측치를 0으로 채우기
    lag_columns = [f'{col}_lag_{i}' for col in target_cols for i in range(1, 5)]
    result_df[lag_columns] = result_df[lag_columns].fillna(0).astype('float32')
    
    return result_df

def create_rolling_features(df: pd.DataFrame, target_cols: list, windows: list = [2, 3]) -> pd.DataFrame:
    """
    지정된 컬럼들에 대해 rolling statistics를 생성하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    target_cols : list
        rolling statistics를 생성할 대상 컬럼들의 리스트
    windows : list
        rolling window 크기 리스트
    
    Returns:
    --------
    pandas.DataFrame
        rolling statistics가 추가된 데이터프레임
    """
    result_df = df.copy()
    result_df = result_df.sort_values(['PIDWON', 'YEAR'])
    
    for col in target_cols:
        for window in windows:
            # 이동 평균
            result_df[f'{col}_rolling_mean_{window}'] = result_df.groupby('PIDWON')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().astype('float32')
            )
            # 이동 표준편차
            result_df[f'{col}_rolling_std_{window}'] = result_df.groupby('PIDWON')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().astype('float32')
            )
            # 이동 최대값
            result_df[f'{col}_rolling_max_{window}'] = result_df.groupby('PIDWON')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max().astype('float32')
            )
            # 이동 최소값
            result_df[f'{col}_rolling_min_{window}'] = result_df.groupby('PIDWON')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min().astype('float32')
            )
    
    return result_df

def create_cumulative_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """
    지정된 컬럼들에 대해 cumulative statistics를 생성하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    target_cols : list
        cumulative statistics를 생성할 대상 컬럼들의 리스트
    
    Returns:
    --------
    pandas.DataFrame
        cumulative statistics가 추가된 데이터프레임
    """
    result_df = df.copy()
    result_df = result_df.sort_values(['PIDWON', 'YEAR'])
    
    for col in target_cols:
        # 누적 합계
        result_df[f'{col}_cumsum'] = result_df.groupby('PIDWON')[col].cumsum().astype('float32')
        # 누적 평균
        result_df[f'{col}_cummean'] = result_df.groupby('PIDWON')[col].transform(
            lambda x: x.expanding().mean()
        ).astype('float32')
        # 누적 표준편차
        result_df[f'{col}_cumstd'] = result_df.groupby('PIDWON')[col].transform(
            lambda x: x.expanding().std()
        ).astype('float32')
    
    return result_df

def create_yoy_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """
    지정된 컬럼들에 대해 전년 대비 변화율을 계산하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    target_cols : list
        변화율을 계산할 대상 컬럼들의 리스트
    
    Returns:
    --------
    pandas.DataFrame
        변화율이 추가된 데이터프레임
    """
    result_df = df.copy()
    result_df = result_df.sort_values(['PIDWON', 'YEAR'])
    
    for col in target_cols:
        # 전년 대비 절대적 변화
        result_df[f'{col}_yoy_diff'] = result_df.groupby('PIDWON')[col].diff().astype('float32')
        # 전년 대비 변화율 (%)
        result_df[f'{col}_yoy_pct'] = result_df.groupby('PIDWON')[col].pct_change() * 100
    
    return result_df

#############################################
# 6. 모델 평가 및 예측 함수들
#############################################
def apply_lasso_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 0.01):
    # LASSO 모델 학습
    lasso = Lasso(alpha=0.01)  # alpha 값은 조정 가능
    lasso.fit(X_train, y_train)

    # 특성 중요도 계산
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': lasso.coef_
    })

    # 계수 절대값 기준으로 정렬
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

    # 결과 출력
    print("LASSO Feature Selection Results:")
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    # 시각화
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(20), 
                x='Coefficient', 
                y='Feature')
    plt.title('Top 20 Features by LASSO Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # 제거된 특성 확인 (계수가 0인 특성)
    zero_features = feature_importance[feature_importance['Coefficient'] == 0]
    print("\nFeatures removed by LASSO (coefficient = 0):")
    print(f"Number of removed features: {len(zero_features)}")
    print(zero_features['Feature'].tolist())

    return zero_features['Feature'].tolist()




def predict_model(X_test_indexed: pd.DataFrame, X_test: pd.DataFrame, models: list, log_transformed: bool = False) -> list:
    """
    여러 모델의 예측을 수행하는 함수
    
    Parameters:
    -----------
    X_test_indexed : pandas.DataFrame
        인덱스가 있는 테스트 데이터
    X_test : pandas.DataFrame
        테스트 데이터
    models : list
        예측에 사용할 모델 리스트
    log_transformed : bool
        로그 변환 여부
    
    Returns:
    --------
    list
        각 모델의 예측값 리스트
    """
    fe_results, lr, xgb, lgb, cat, rf = models
    used_cols = fe_results.model.exog.vars # 모델 학습 시 실제 사용된 변수들만 추출
    X_test_aligned = X_test_indexed[used_cols] # 테스트 데이터에 동일한 컬럼 적용

    fe_pred = fe_results.predict(X_test_aligned).squeeze()
    lr_pred = lr.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    lgb_pred = lgb.predict(X_test)
    cat_pred = cat.predict(X_test)
    rf_pred = rf.predict(X_test)

    if log_transformed:
        results = [np.exp(fe_pred), np.exp(lr_pred), np.exp(xgb_pred), np.exp(lgb_pred), np.exp(cat_pred), np.exp(rf_pred)]
    else:
        results = [fe_pred, lr_pred, xgb_pred, lgb_pred, cat_pred, rf_pred]
    return results

def transform_y(y_test: np.ndarray, log_transformed: bool = False) -> np.ndarray:
    """
    타겟 변수를 변환하는 함수
    
    Parameters:
    -----------
    y_test : array-like
        변환할 타겟 변수
    log_transformed : bool
        로그 변환 여부
    
    Returns:
    --------
    array-like
        변환된 타겟 변수
    """
    if log_transformed:
        return np.exp(y_test)
    else:
        return y_test

def calculate_adjusted_r2(r2: float, n: int, p: int) -> float:
    """
    Adjusted R²를 계산하는 함수
    
    Parameters:
    -----------
    r2 : float
        R² 값
    n : int
        샘플 수
    p : int
        특성 수
    
    Returns:
    --------
    float
        Adjusted R² 값
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE(Mean Absolute Percentage Error)를 계산하는 함수
    
    Parameters:
    -----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값
    
    Returns:
    --------
    float
        MAPE 값
    """
    # 실제값이 0이거나 음수인 경우 제외
    mask = y_true > 0
    if not np.any(mask):
        return np.nan
    
    # 양수인 실제값에 대해서만 MAPE 계산
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, X_test: pd.DataFrame) -> tuple:
    """
    모델의 성능을 평가하는 함수
    
    Parameters:
    -----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값
    model_name : str
        모델 이름
    X_test : pandas.DataFrame
        테스트 데이터
    
    Returns:
    --------
    tuple
        (mse, rmse, mae, mape, r2, adj_r2)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    # Adjusted R² 계산
    n = len(y_true)  # 샘플 수
    p = X_test.shape[1]  # 특성 수
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    
    print(f"\n{model_name} 모델 성능 평가:")
    print(f"RMSE: {rmse:,.1f}")
    print(f"MAE: {mae:,.1f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Adjusted R2 Score: {adj_r2:.3f}")
    
    return mse, rmse, mae, mape, r2, adj_r2


def analyze_feature_importance(model: object, feature_names: list, top_n: int = 20, 
                               figsize: tuple = (12, 8), type: str = 'xgb') -> pd.DataFrame:
    """
    RandomForest 모델의 Feature Importance를 분석하고 시각화하는 함수
    
    Parameters:
    -----------
    model : RandomForestRegressor
        학습된 RandomForest 모델
    feature_names : list
        특성 이름 리스트
    top_n : int
        상위 몇 개의 특성을 보여줄지 지정 (기본값: 20)
    figsize : tuple
        그래프 크기 (기본값: (12, 8))
    
    Returns:
    --------
    pandas.DataFrame
        Feature Importance 분석 결과
    """
    # Feature Importance 추출
    importances = model.feature_importances_
    
    # 특성 중요도와 이름을 데이터프레임으로 변환
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # 중요도 기준으로 정렬
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # 상위 N개 특성만 선택
    top_features = feature_importance.head(top_n)
    
    # 시각화
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importance', pad=20)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # 중요도가 0인 특성 개수 출력
    zero_importance = feature_importance[feature_importance['importance'] == 0]
    print(f"\n중요도가 0인 특성 개수: {len(zero_importance)}")
    
    # 중요도 통계 출력
    print("\nFeature Importance 통계:")
    print(feature_importance['importance'].describe())
    
    # 중요도 기준으로 정렬된 전체 특성 목록 반환
    return feature_importance

def handle_abnormal_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
                        

def analyze_linear_coefficients(model, feature_names, top_n=20, figsize=(12, 8), title_prefix="Linear Regression"):
    """
    선형 회귀 모델의 회귀 계수를 분석하고 시각화하는 함수

    Parameters:
    -----------
    model : sklearn.linear_model.LinearRegression or statsmodels 결과 객체
        학습된 선형 회귀 모델
    feature_names : list
        특성 이름 리스트
    top_n : int
        절댓값 기준 상위 N개의 계수를 시각화 (기본값: 20)
    figsize : tuple
        그래프 크기
    title_prefix : str
        그래프 제목 앞에 붙일 텍스트

    Returns:
    --------
    pandas.DataFrame
        회귀 계수 정리된 데이터프레임
    """
    # 계수 추출
    if hasattr(model, 'coef_'):
        # sklearn 계열
        coefs = model.coef_
    elif hasattr(model, 'params'):
        # statsmodels 계열
        coefs = model.params.values
    else:
        raise ValueError("모델 객체가 회귀 계수를 제공하지 않습니다.")

    # 데이터프레임으로 구성
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs
    })
    
    # 절댓값 기준 상위 N개 추출
    top_coef = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index).head(top_n)

    # 시각화
    plt.figure(figsize=figsize)
    sns.barplot(x='coefficient', y='feature', data=top_coef, palette='coolwarm')
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f'{title_prefix}: Top {top_n} Coefficients', pad=20)
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # 계수 통계 출력
    print("\nCoefficient 통계:")
    print(coef_df['coefficient'].describe())
    
    # 전체 계수 정렬 반환
    return coef_df.sort_values('coefficient', key=abs, ascending=False)


def calculate_pi(model, X_train, y_train, X_test, alpha=0.05):
    """
    예측값의 Prediction Interval (PI)를 계산하는 함수
    
    Parameters:
    -----------
    model : fitted model
        학습된 모델 (fit 완료)
    X_train : pd.DataFrame
        학습 입력 데이터
    y_train : pd.Series or np.array
        학습 정답값
    X_test : pd.DataFrame
        테스트 입력 데이터
    alpha : float
        유의수준 (기본값: 0.05 → 95% PI)
        
    Returns:
    --------
    tuple : (예측값, 하한값, 상한값)
        각 테스트 샘플에 대한 예측값과 95% Prediction Interval
    """
    # 1. 테스트 예측값
    y_pred = model.predict(X_test)
    
    # 2. 학습 예측값과 잔차
    y_train_pred = model.predict(X_train)
    residuals = y_train - y_train_pred
    
    # 3. 잔차의 표준편차
    resid_std = np.std(residuals)
    
    # 4. z-score (95% 기준)
    z = norm.ppf(1 - alpha/2)  # e.g., 1.96 for 95%
    
    # 5. Prediction Interval
    pi_lower = y_pred - z * resid_std
    pi_upper = y_pred + z * resid_std
    
    return y_pred, pi_lower, pi_upper



# # 예측값 앙상블
# ensemble_pred = (rf_pred + lgb_pred + xgb_pred) / 3

# 부트스트래핑 기반 앙상블 Prediction Interval
def calculate_ensemble_pi(models: dict, X_test: pd.DataFrame, n_bootstrap: int = 1000, alpha: float = 0.05):
    preds = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_sample = X_test.iloc[idx]

        # 각 모델의 예측 평균
        pred_rf = models['rf'].predict(X_sample)
        pred_lgb = models['lgb'].predict(X_sample)
        pred_xgb = models['xgb'].predict(X_sample)

        ensemble_pred = (pred_rf + pred_lgb + pred_xgb) / 3
        preds.append(ensemble_pred)

    preds = np.array(preds)

    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    point = ((pred_rf + pred_lgb + pred_xgb) / 3)

    return point, lower, upper


def plot_prediction_pi(results: pd.DataFrame, index: int, figsize=(12, 6)):
    """
    특정 인덱스의 예측값과 Prediction Interval을 수평 방향으로 시각화하는 함수
    """

    # 모델별 색상 정의
    colors = {
        'Linear Regression': '#1f77b4',  # 파란색
        'RandomForest': '#ff7f0e',       # 주황색
        'XGBoost': '#2ca02c',            # 초록색
        'LightGBM': '#d62728',           # 빨간색
        # 'Ensemble': '#9467bd'            # 보라색
    }

    # 모델별 접두사
    model_prefixes = {
        'Linear Regression': 'lr',
        'RandomForest': 'rf',
        'XGBoost': 'xgb',
        'LightGBM': 'lgb',
        # 'Ensemble': 'ensemble'
    }

    # 그래프 생성
    plt.figure(figsize=figsize)
    y_positions = np.arange(len(model_prefixes))

    for i, (model_name, prefix) in enumerate(model_prefixes.items()):
        lower = results.loc[index, f'{prefix}_lower_pi']
        upper = results.loc[index, f'{prefix}_upper_pi']
        pred = results.loc[index, f'{prefix}_prediction']

        # 수평 Prediction Interval
        plt.hlines(
            y=i, xmin=lower, xmax=upper,
            colors=colors[model_name], alpha=0.4, linewidth=8,
            label=f'{model_name} PI'
        )

        # 예측값 점
        plt.scatter(
            pred, i,
            color=colors[model_name],
            s=100,
            edgecolor='black',
            label=f'{model_name} Prediction'
        )

    # 실제값 수직선
    plt.axvline(
        x=results.loc[index, 'actual'],
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Actual Value'
    )

    # 꾸미기
    plt.title(f'Predictions and 95% Prediction Intervals for Index {index}', pad=20)
    plt.xlabel('Medical Expense')
    plt.ylabel('Models')
    plt.yticks(y_positions, model_prefixes.keys())
    plt.grid(True, axis='x', alpha=0.3)
    plt.xlim(left=0)

    # 범례 (중복 제거 없이 전체 표시)
    handles, labels = plt.gca().get_legend_handles_labels()
    combined = list(zip(labels, handles))
    seen = set()
    unique = [(l, h) for l, h in combined if not (l in seen or seen.add(l))]
    plt.legend([h for l, h in unique], [l for l, h in unique], bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def calculate_pi_statistics(results: pd.DataFrame):
    """
    각 모델의 Prediction Interval 통계를 계산하는 함수
    
    Parameters:
    -----------
    results : DataFrame
        예측값과 Prediction Interval이 포함된 데이터프레임
    
    Returns:
    --------
    tuple
        (통계 데이터프레임, 시각화용 데이터프레임)
    """
    # 모델별 컬럼 접두사
    model_prefixes = {
        'Linear Regression': 'lr',
        'RandomForest': 'rf',
        'XGBoost': 'xgb',
        'LightGBM': 'lgb'
    }
    
    # 통계 계산을 위한 딕셔너리
    pi_stats = {}
    
    # 시각화용 데이터 준비
    pi_widths = []
    model_names = []
    
    for model_name, prefix in model_prefixes.items():
        # PI 너비 계산
        pi_width = results[f'{prefix}_upper_pi'] - results[f'{prefix}_lower_pi']
        
        # 통계 계산
        pi_stats[model_name] = {
            'Mean PI Width': round(pi_width.mean(), 2),
            'Std PI Width': round(pi_width.std(), 2),
            'Min PI Width': round(pi_width.min(), 2),
            'Max PI Width': round(pi_width.max(), 2),
            'Median PI Width': round(pi_width.median(), 2)
        }
        
        # 시각화용 데이터 추가
        pi_widths.extend(pi_width)
        model_names.extend([model_name] * len(pi_width))
    
    # 통계 데이터프레임 생성
    stats_df = pd.DataFrame(pi_stats).T
    stats_df = stats_df[['Mean PI Width', 'Std PI Width', 'Min PI Width', 'Max PI Width', 'Median PI Width']]
    
    # 시각화용 데이터프레임 생성
    plot_df = pd.DataFrame({
        'Model': model_names,
        'PI Width': pi_widths
    })
    
    return stats_df, plot_df




