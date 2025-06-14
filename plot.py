import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import random
random.seed(42)
np.random.seed(42)

id_columns = ['HHIDWON', 'PIDWON']
plot_path = 'plots'

#############################################
# 1. 기본 분포 시각화 함수들
#############################################

def draw_histogram(df: pd.DataFrame, continuous_columns: list[str]) -> None:
    """
    연속형 변수들의 히스토그램을 그리는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    continuous_columns : list[str]
        히스토그램을 그릴 연속형 변수 컬럼 리스트
    
    Returns:
    --------
    None
        히스토그램을 저장하고 표시
    """
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
    plt.savefig(f'{plot_path}/histogram.png')
    plt.show()

def draw_countplot(df: pd.DataFrame, categorical_columns: list[str]) -> None:
    """
    범주형 변수들의 카운트플롯을 그리는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    categorical_columns : list[str]
        카운트플롯을 그릴 범주형 변수 컬럼 리스트
    
    Returns:
    --------
    None
        카운트플롯을 저장하고 표시
    """
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
    plt.savefig(f'{plot_path}/countplot.png')
    plt.show()

def draw_kde_plot(df: pd.DataFrame, numeric_continuous_cols: list[str]) -> None:
    """
    연속형 변수들의 KDE 플롯을 그리는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    numeric_continuous_cols : list[str]
        KDE 플롯을 그릴 연속형 변수 컬럼 리스트
    
    Returns:
    --------
    None
        KDE 플롯을 저장하고 표시
    """
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
    plt.savefig(f'{plot_path}/kde.png')
    plt.show()

#############################################
# 2. 관계 분석 시각화 함수들
#############################################

def draw_scatterplot(df: pd.DataFrame, continuous_columns: list[str]) -> None:
    """
    연속형 변수들과 타겟 변수 간의 산점도를 그리는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    continuous_columns : list[str]
        산점도를 그릴 연속형 변수 컬럼 리스트
    
    Returns:
    --------
    None
        산점도를 저장하고 표시
    """
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
    plt.savefig(f'{plot_path}/scatterplot.png')
    plt.show()

def draw_boxplot(df: pd.DataFrame, columns: list[str], type: str = 'ordinal') -> None:
    """
    범주형 변수들과 타겟 변수 간의 박스플롯을 그리는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    columns : list[str]
        박스플롯을 그릴 범주형 변수 컬럼 리스트
    type : str
        변수 유형 ('ordinal' 또는 'nominal')
    
    Returns:
    --------
    None
        박스플롯을 저장하고 표시
    """
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

    plt.savefig(f'{plot_path}/boxplot_{type}.png')
    plt.show()

def plot_correlation_matrix(data: pd.DataFrame, analysis_cols: list[str], 
                          target_col: str = 'medical_expense', method: str = 'spearman', 
                          type: str = 'continuous') -> None:
    """
    변수들 간의 상관관계 행렬을 히트맵으로 시각화하는 함수
    
    Parameters:
    -----------
    data : pandas.DataFrame
        입력 데이터프레임
    analysis_cols : list[str]
        분석할 변수 컬럼 리스트
    target_col : str
        타겟 변수명
    method : str
        상관계수 계산 방법 ('spearman' 또는 'pearson')
    type : str
        변수 유형 ('continuous' 또는 'ordinal')
    
    Returns:
    --------
    None
        상관관계 히트맵을 저장하고 표시
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = data[analysis_cols + [target_col]].corr(method)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})
    title = f'{method.upper()} Correlation Heatmap of {", ".join(analysis_cols)} vs {target_col}'
    plt.title(f'Spearman Correlation Heatmap of {type.upper()} Variables', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/correlation_heatmap_{type}.png')
    plt.show()

def visualize_categorical_relationships(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 변수들과 타겟 변수 간의 관계를 히트맵으로 시각화하는 함수
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        범주형 변수 분석 결과 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        상관관계 분석 결과를 정렬한 데이터프레임
    """
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

#############################################
# 3. 모델 해석 시각화 함수들
#############################################

def implement_lime(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, sample_idx: int) -> tuple:
    """
    LIME을 사용하여 모델의 예측을 설명하는 함수
    
    Parameters:
    -----------
    model : object
        학습된 모델
    X_train : pandas.DataFrame
        학습 데이터
    X_test : pandas.DataFrame
        테스트 데이터
    sample_idx : int
        설명할 샘플의 인덱스
    
    Returns:
    --------
    tuple
        (LIME explainer, LIME explanation)
    """
    X_train_sample = X_train.copy()
    X_test_sample = X_test.copy()

    X_train_array = X_train_sample.values
    X_test_array = X_test_sample.values

    # LIME Explainer
    explainer = LimeTabularExplainer(
        training_data=X_train_array,
        feature_names=X_train_sample.columns,
        mode='regression'  # 회귀 모드
    )

    # shape를 (1, n_features)로 조정
    instance = X_test_array[sample_idx].reshape(1, -1)

    # LIME 설명 생성
    exp = explainer.explain_instance(
        data_row=instance[0],  # 1차원 배열로 변환
        predict_fn=model.predict,      # 회귀 예측 함수
        num_features=10                # 설명에 표시할 최대 Feature 수
    )

    # LIME 결과 시각화
    exp.show_in_notebook(show_table=True)
    return explainer, exp

def plot_lime(exp: object) -> None:
    """
    LIME 설명 결과를 시각화하는 함수
    
    Parameters:
    -----------
    exp : object
        LIME explanation 객체
    
    Returns:
    --------
    None
        LIME 설명 시각화를 표시
    """
    # 피처 리스트 및 그에 해당하는 가중치
    features = exp.as_list()

    # 가중치 값에 따라 피처 정렬
    features.sort(key=lambda x: x[1], reverse=True)

    # 두 리스트로 나누기
    feature_names, feature_weights = zip(*features)

    # 바 플랏 그리기
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_weights, color='skyblue')
    plt.xlabel('Feature Weights')
    plt.gca().invert_yaxis()
    plt.show()

def explain_instance_text(explainer: object, instance: object, predict_fn: object, top_k: int = 5) -> list[str]:
    """
    LIME 설명을 텍스트로 변환하는 함수
    
    Parameters:
    -----------
    explainer : object
        LIME explainer 객체
    instance : object
        설명할 인스턴스
    predict_fn : object
        예측 함수
    top_k : int
        상위 몇 개의 특성을 설명할지 지정
    
    Returns:
    --------
    list[str]
        특성별 설명 텍스트 리스트
    """
    exp = explainer.explain_instance(instance, predict_fn=predict_fn, num_features=top_k)
    features = exp.as_list()

    explanations = []
    for condition, weight in features:
        effect = "증가시키는" if weight > 0 else "감소시키는"
        # 곱셈적 영향력으로 해석
        multiplier = np.exp(weight)
        percentage = (multiplier - 1) * 100  # 백분율 변화로 변환
        explanations.append(
            f"- 조건 [{condition}] 은(는) 예측값을 약 {percentage:.1f}% {effect} 데 기여했습니다."
        )
    return explanations

def plot_pi_distribution(plot_df: pd.DataFrame) -> plt.Figure:
    """
    Prediction Interval 분포를 수평 violin plot으로 시각화하는 함수
    
    Parameters:
    -----------
    plot_df : pandas.DataFrame
        시각화용 데이터프레임 (Model, PI Width 컬럼 포함)
    
    Returns:
    --------
    matplotlib.pyplot.Figure
        수평 violin plot이 그려진 그래프
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_df, y='Model', x='PI Width', orient='h')
    
    # 그래프 꾸미기
    plt.title('Distribution of Prediction Interval Widths by Model', pad=20)
    plt.ylabel('Models')
    plt.xlabel('Width of Prediction Intervals')
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/pi_distribution.png')
    
    return plt