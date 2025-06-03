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


def remove_outliers_iqr(df, column='medical_expense', lower_bound=0.25, upper_bound=0.75, multiplier=1.5):
    """
    IQR 방법을 사용하여 이상치를 제거하는 함수
    
    Parameters:
    -----------
    df : pandas DataFrame
        입력 데이터프레임
    column : str
        이상치를 제거할 컬럼명 (기본값: 'TOT_EXP')
    lower_bound : float
        하위 사분위수 경계 (기본값: 0.25)
    upper_bound : float
        상위 사분위수 경계 (기본값: 0.75)
    multiplier : float
        IQR 곱셈 계수 (기본값: 1.5)
    
    Returns:
    --------
    tuple
        (이상치가 제거된 데이터프레임, 제거된 이상치의 개수, 이상치의 비율)
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

def winsorize_data(df, column='medical_expense', lower_percentile=5, upper_percentile=95):
    """
    Winsorization을 수행하는 함수
    
    Parameters:
    -----------
    df : pandas DataFrame
        입력 데이터프레임
    column : str
        Winsorization을 수행할 컬럼명
    lower_percentile : float
        하위 경계 백분위수 (기본값: 5)
    upper_percentile : float
        상위 경계 백분위수 (기본값: 95)
    
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


def predict_model(X_test_indexed: pd.DataFrame, X_test: pd.DataFrame, models: list, log_transformed: bool = False):
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

def transform_y(y_test, log_transformed = False):
    if log_transformed:
        return np.exp(y_test)
    else:
        return y_test

# Adjusted R² 계산 함수
def calculate_adjusted_r2(r2, n, p):
    """
    n: 샘플 수
    p: 특성 수
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def calculate_mape(y_true, y_pred):
    """
    MAPE(Mean Absolute Percentage Error) 계산
    수정된 버전: 0이나 음수 값 처리
    """
    # 실제값이 0이거나 음수인 경우 제외
    mask = y_true > 0
    if not np.any(mask):
        return np.nan
    
    # 양수인 실제값에 대해서만 MAPE 계산
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

# 성능 평가를 위한 함수
def evaluate_model(y_true, y_pred, model_name, X_test):
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


def analyze_feature_importance(model, feature_names, top_n=20, figsize=(12, 8)):
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