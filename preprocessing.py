import pandas as pd
import numpy as np
import duckdb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

id_columns = ['HHIDWON', 'PIDWON']
idx_columns = ['HHIDWON', 'PIDWON', 'YEAR']

def filter_cd(df_cd: pd.DataFrame) -> pd.DataFrame:
    # ROW (당뇨병 환자) 필터링
    filtered_df_cd = df_cd[(df_cd['CD3_2'] == 1) & (df_cd['CDNUM'] == 2)]#.reset_index()
    filtered_df_cd.index = [i for i in range(0, len(filtered_df_cd))]

    unique_hhid = filtered_df_cd['HHIDWON'].unique() # 당뇨병 진단받은 환자들의 가구 ID
    unique_pid = filtered_df_cd['PIDWON'].unique() # 당뇨병 진단받은 환자들의 ID. / PIDWON = HHIDWON + PID

    # 필터링 여부 재확인
    print(f"CD3_2(의사진단여부) 컬럼 내 unique 값 = {filtered_df_cd['CD3_2'].unique()}")
    print(f"CDNUM(만성질환번호) 컬럼 내 unique 값 = {filtered_df_cd['CDNUM'].unique()}")
    print(f"필터링 된 데이터셋 내의 unique 가구 ID의 수 = {len(unique_hhid)}")
    print(f"필터링 된 데이터셋 내의 unique 가구원 ID의 수 = {len(unique_pid)}")
    print(f"필터링 이전 대비 이후 환자 데이터셋 크기 비율 = {(len(filtered_df_cd) / len(df_cd)):.4f}")
    print(f"필터링 이전 대비 이후 환자 ID 비율 = {(len(unique_pid) / df_cd['PIDWON'].nunique()):.4f}")

    return filtered_df_cd, unique_hhid, unique_pid

def filter_rest(all_df: pd.DataFrame, unique_hhid: list, unique_pid: list) -> pd.DataFrame:
    filtered_df = all_df[all_df['PIDWON'].isin(unique_pid)]

    print(f"필터링 이전 대비 이후 환자 데이터셋 크기 비율 = {(len(filtered_df) / len(all_df)):.4f}")
    print(f"필터링 이전 대비 이후 환자 ID 비율 = {(len(unique_pid) / all_df['PIDWON'].nunique()):.4f}")

    return filtered_df

def filter_hh(all_df: pd.DataFrame, unique_hhid: list) -> pd.DataFrame:
    filtered_df = all_df[all_df['HHIDWON'].isin(unique_hhid)]

    print(f"필터링 된 데이터셋 내의 unique 가구 ID의 수 = {len(unique_hhid)}")
    print(f"필터링 이전 대비 이후 환자 데이터셋 크기 비율 = {(len(filtered_df) / len(all_df)):.4f}")
    return filtered_df

def map_df(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    for col, mapper in mapping_dict.items():
        # 매핑 전 원본 값 확인
        original_values = set(df[col].unique())
        
        # 매핑 수행
        mapped_values = df[col].map(mapper)
        
        # 매핑된 값이 -1, 0, 1만 포함하는지 확인
        if set(mapped_values.unique()).issubset({-1, 0, 1}):
            df[col] = mapped_values.astype('int')
        else:
            df[col] = mapped_values.astype('category')
    return df

def rename_df(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    df.rename(columns=rename_dict, inplace=True)
    return df

def onehot_df(df: pd.DataFrame, onehot_columns: list) -> pd.DataFrame:
    # 원-핫 인코딩 수행
    df_dummies = pd.get_dummies(df, columns=onehot_columns)
    
    # 원-핫 인코딩된 컬럼들을 정수형으로 변환 (-1, 0, 1 가능)
    for col in df_dummies.columns:
        if col not in df.columns:  # 새로 생성된 원-핫 인코딩 컬럼만 처리
            df_dummies[col] = df_dummies[col].astype('int8')  # -1, 0, 1을 저장할 수 있는 int8 사용
    
    return df_dummies

def convert_into_int(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        # # 원-핫 인코딩 컬럼인지 확인 (0과 1만 있는 컬럼)
        # is_one_hot = df[col].nunique() <= 2 and set(df[col].unique()).issubset({0, 1})
        
        # 범주형 변수인 경우 (object, category, bool, 원-핫 인코딩)
        if df[col].dtype in ['object', 'category', 'bool']: # or is_one_hot:
            df[col] = df[col].astype('category')
        # 불리언 변수인 경우 (True/False를 1/0으로 변환하면서 category 유지)
        # elif df[col].dtype == 'bool':
        #     df[col] = df[col].astype(int).astype('category')
        # 연속형 변수인 경우 (float64, int64)
        elif df[col].dtype in ['float64', 'int64', 'int8']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        # 나머지 타입은 그대로 유지
        else:
            continue
    return df

def move_year_column(df: pd.DataFrame) -> pd.DataFrame:
    # YEAR 컬럼을 가장 오른쪽으로 이동
    year_col = df.pop('YEAR')  # YEAR 컬럼 제거하고 저장
    df['YEAR'] = year_col      # 맨 뒤에 추가
    return df

def move_y_column(df: pd.DataFrame) -> pd.DataFrame:
    # YEAR 컬럼을 가장 오른쪽으로 이동
    y_col = df.pop('medical_expense')  # medical_expense 컬럼 제거하고 저장
    df['medical_expense'] = y_col      # 맨 뒤에 추가
    return df

def calculate_num_years(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    for col in target_cols:
        # YEAR 컬럼을 정수형으로 변환
        year_col = pd.to_numeric(df['YEAR'], errors='coerce')
        
        # 대상 컬럼을 정수형으로 변환
        target_col = pd.to_numeric(df[col], errors='coerce')
        
        # 각 행별로 처리
        df[col] = df.apply(
            lambda row: 0 if pd.isnull(row[col]) or row[col] == -1 
            else int(row['YEAR'] - row[col]), 
            axis=1
        )
    return df

def convert_won(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    for col in target_cols:
        df[col] *= 10000
    return df

def map_dataframes(df_cd: pd.DataFrame, df_ind: pd.DataFrame, df_hh: pd.DataFrame, 
                   df_er: pd.DataFrame, df_ou: pd.DataFrame, df_appen: pd.DataFrame, 
                   df_in: pd.DataFrame) -> pd.DataFrame:
    """
    여러 데이터프레임을 PIDWON과 YEAR를 기준으로 매핑합니다.
    final_df_hh는 HHIDWON과 YEAR를 기준으로 매핑됩니다.
    
    Parameters:
    -----------
    final_df_cd : pandas.DataFrame
        기준이 되는 데이터프레임
    final_df_ind : pandas.DataFrame
        개인 정보 데이터프레임
    final_df_hh : pandas.DataFrame
        가구 정보 데이터프레임
    final_df_er : pandas.DataFrame
        응급실 데이터프레임
    final_df_ou : pandas.DataFrame
        외래진료 데이터프레임
    final_df_appen : pandas.DataFrame
        부가진료 데이터프레임
    final_df_in : pandas.DataFrame
        입원 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        매핑된 최종 데이터프레임
    """
    # DuckDB 연결 생성
    con = duckdb.connect(database=':memory:')
    
    # 데이터프레임들을 DuckDB 테이블로 등록
    con.register('cd', df_cd)
    con.register('ind', df_ind)
    con.register('hh', df_hh)
    con.register('er', df_er)
    con.register('ou', df_ou)
    con.register('appen', df_appen)
    con.register('inpatient', df_in)
    
    # SQL 쿼리 작성
    query = """
    SELECT 
        cd.HHIDWON,
        cd.PIDWON,
        cd.YEAR,
        cd.* EXCLUDE (HHIDWON, PIDWON, YEAR),
        ind.* EXCLUDE (HHIDWON, PIDWON, YEAR),
        hh.* EXCLUDE (HHIDWON, YEAR),
        er.* EXCLUDE (HHIDWON, PIDWON, YEAR),
        ou.* EXCLUDE (HHIDWON, PIDWON, YEAR),
        appen.* EXCLUDE (HHIDWON, PIDWON, YEAR),
        inpatient.* EXCLUDE (HHIDWON, PIDWON, YEAR)
    FROM cd
    LEFT JOIN ind ON cd.PIDWON = ind.PIDWON AND cd.YEAR = ind.YEAR
    LEFT JOIN hh ON cd.HHIDWON = hh.HHIDWON AND cd.YEAR = hh.YEAR
    LEFT JOIN er ON cd.PIDWON = er.PIDWON AND cd.YEAR = er.YEAR
    LEFT JOIN ou ON cd.PIDWON = ou.PIDWON AND cd.YEAR = ou.YEAR
    LEFT JOIN appen ON cd.PIDWON = appen.PIDWON AND cd.YEAR = appen.YEAR
    LEFT JOIN inpatient ON cd.PIDWON = inpatient.PIDWON AND cd.YEAR = inpatient.YEAR
    """
    
    # 쿼리 실행 및 결과를 pandas DataFrame으로 변환
    result_df = con.execute(query).df()
    
    # 연결 종료
    con.close()
    
    return result_df


def fill_na_vals(df:pd.DataFrame, target_cols:list[str], constant_cols:list[str], user_cols:list[str], incremental_cols:list[str],
                 categorical_nominal_cols:list[str], categorical_ordinal_cols:list[str], numeric_continuous_cols:list[str], numeric_discrete_cols:list[str]) -> pd.DataFrame:
    
    clean_df = df.copy()
    for col in target_cols:
        # print(col)
        if col in constant_cols:  # 수치형 - 상수형 변수는 해당 가구원의 중앙값으로 대체
            # PIDWON으로 그룹화하여 각 가구원별 중앙값으로 채우기
            # 중앙값을 계산할 때 정수형으로 변환
            clean_df[col] = clean_df.groupby('PIDWON')[col].transform(
                lambda x: x.fillna(int(x.median())) if x.dtype == 'Int64' else x.fillna(x.median())
            )
        elif col in user_cols:  # 가구원 고유의 변수 (키, 몸무게, bmi)는 이전 데이터 값으로 대체
            # PIDWON으로 그룹화하여 이전 값으로 채우기
            temp_forward = clean_df.groupby('PIDWON')[col].fillna(method='ffill')
            # 이전 값이 없는 경우에만 다음 값으로 채우기
            clean_df[col] = temp_forward.fillna(clean_df.groupby('PIDWON')[col].fillna(method='bfill'))
            # 그럼에도 값이 없는 경우에는 -1로 대체
            clean_df[col] = clean_df[col].fillna(-1)
            
        elif col in incremental_cols:  # 증가형 변수는 해당 가구원의 작년도 값 + 1 OR 내년도 값 - 1 로 대체
            # 이전 값과 다음 값 가져오기
            prev_val = clean_df.groupby('PIDWON')[col].fillna(method='ffill')
            next_val = clean_df.groupby('PIDWON')[col].fillna(method='bfill')
            
            # 이전 값이 -1이 아닌 경우에만 +1 적용
            temp_forward = prev_val.where(prev_val != -1, -1)
            temp_forward = temp_forward.where(temp_forward == -1, temp_forward + 1)
            
            # 다음 값이 -1이 아닌 경우에만 -1 적용
            temp_backward = next_val.where(next_val != -1, -1)
            temp_backward = temp_backward.where(temp_backward == -1, temp_backward - 1)
            
            # 이전 값으로 채우기 시도
            clean_df[col] = temp_forward
            
            # 이전 값이 없는 경우에만 다음 값으로 채우기
            clean_df[col] = clean_df[col].fillna(temp_backward)
            
            # 그럼에도 값이 없는 경우에는 -1로 대체
            clean_df[col] = clean_df[col].fillna(-1)
            
        elif col in numeric_continuous_cols and col not in [user_cols, incremental_cols, constant_cols]:  # 수치형 - 연속형 변수는 해당 PIDWON의 중앙값으로 대체
            # PIDWON으로 그룹화하여 각 가구원별 중앙값으로 채우기
            # clean_df[col] = clean_df.groupby('PIDWON')[col].transform(
            #     lambda x: x.fillna(round(x.dropna().median())) if x.dtype == 'Int64' and not x.dropna().empty else x.fillna(0)
            # )
            clean_df[col] = clean_df[col].fillna(0)

        elif col in numeric_discrete_cols and col not in [user_cols, incremental_cols, constant_cols]: # 수치형 - 이산형 변수는 0으로 대체 (주로 원핫인코딩된 변수)
            clean_df[col] = clean_df[col].fillna(0)
            
        elif col in categorical_nominal_cols:  # 범주형 - 명목형 변수는 해당 가구원의 최빈값으로 대체
            # PIDWON으로 그룹화하여 각 가구원별 최빈값으로 채우기
            clean_df[col] = clean_df.groupby('PIDWON')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mode()))
            # 그럼에도 결측치가 있는 경우에는, -1로 대체
            clean_df[col] = clean_df[col].fillna(-1)
            
        elif col in categorical_ordinal_cols:  # 범주형 - 순서형 변수는 0으로 대체
            # PIDWON으로 그룹화하여 각 가구원별 0으로 채우기
            clean_df[col] = clean_df[col].fillna(-1)

    return clean_df


def fill_na_years(df: pd.DataFrame, constant_cols:list[str], user_cols:list[str], incremental_cols:list[str],
                 categorical_nominal_cols:list[str], categorical_ordinal_cols:list[str], numeric_continuous_cols:list[str], numeric_discrete_cols:list[str]) -> pd.DataFrame:
    """
    각 환자(PIDWON)별로 2014~2018년도의 모든 행이 존재하도록 채웁니다.
    없는 연도의 행은 0으로 채워집니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        모든 연도가 채워진 데이터프레임
    """
    # 고유한 (HHIDWON, PIDWON) 쌍 추출
    unique_pairs = df[['HHIDWON', 'PIDWON']].drop_duplicates()
    
    # 모든 연도 리스트
    all_years = [2014, 2015, 2016, 2017, 2018]
    
    # 모든 조합 생성
    full_combinations = []
    for _, row in unique_pairs.iterrows():
        for year in all_years:
            full_combinations.append({
                'HHIDWON': row['HHIDWON'],
                'PIDWON': row['PIDWON'],
                'YEAR': year
            })
    
    # 새로운 데이터프레임 생성
    full_df = pd.DataFrame(full_combinations)
    
    # 원본 데이터프레임과 병합
    result_df = pd.merge(
        full_df,
        df,
        on=['HHIDWON', 'PIDWON', 'YEAR'],
        how='left'
    )
    # print(result_df)

    # result_df_full = fill_na_vals(result_df, list(result_df.columns), constant_cols, user_cols, incremental_cols,
    #              categorical_nominal_cols, categorical_ordinal_cols, numeric_continuous_cols, numeric_discrete_cols)
    
    return result_df


def get_columns_with_missing_values(df: pd.DataFrame) -> list:
    """
    데이터프레임에서 결측치가 하나라도 있는 컬럼들의 리스트를 반환합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    
    Returns:
    --------
    list
        결측치가 있는 컬럼명들의 리스트
    """
    # 각 컬럼별 결측치 개수 계산
    missing_counts = df.isnull().sum()
    
    # 결측치가 하나라도 있는 컬럼만 필터링
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    # 결과 출력
    print(f"총 {len(columns_with_missing)}개의 컬럼에 결측치가 있습니다.")
    for col in columns_with_missing:
        missing_count = missing_counts[col]
        missing_percentage = (missing_count / len(df)) * 100
        print(f"- {col}: {missing_count}개 ({missing_percentage:.2f}%)")
    
    return columns_with_missing


def log_transformation(df, numeric_continuous_cols, method='log1p'):
    """
    수치형 연속형 변수들에 대해 로그 변환을 수행하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    numeric_continuous_cols : list
        로그 변환할 수치형 연속형 변수 리스트
    method : str, default='log1p'
        로그 변환 방법
        - 'log1p': log1p 변환 (log(1+x))
        - 'log': 일반 로그 변환 (log(x))
        - 'log_abs': 절대값 로그 변환 (log(|x|))
    
    Returns:
    --------
    pandas.DataFrame
        로그 변환된 데이터프레임
    """
    # 원본 데이터프레임 복사
    df_transformed = df.copy()
    
    # 변환 전 통계량 저장
    print("=== 변환 전 통계량 ===")
    print(df[numeric_continuous_cols].describe())
    
    # 각 변수별 로그 변환
    for col in numeric_continuous_cols:
        if method == 'log1p':
            # log1p 변환 (음수, 0 값 처리)
            df_transformed[col] = np.log1p(df[col])
            print(f"\n{col} 변환: log1p(x)")
            
        elif method == 'log':
            # 일반 로그 변환 (양수 값만)
            mask = df[col] > 0
            df_transformed[col] = np.log(df[col].where(mask))
            print(f"\n{col} 변환: log(x) (x > 0)")
            
        elif method == 'log_abs':
            # 절대값 로그 변환
            df_transformed[col] = np.log(np.abs(df[col]))
            print(f"\n{col} 변환: log(|x|)")
            
        else:
            raise ValueError("method는 'log1p', 'log', 'log_abs' 중 하나여야 합니다.")
    
    # 변환 후 통계량 출력
    print("\n=== 변환 후 통계량 ===")
    print(df_transformed[numeric_continuous_cols].describe())
    
    # 변환 전후 분포 시각화
    fig, axes = plt.subplots(len(numeric_continuous_cols), 2, figsize=(15, 5*len(numeric_continuous_cols)))
    
    for idx, col in enumerate(numeric_continuous_cols):
        # 변환 전 분포
        sns.histplot(data=df, x=col, ax=axes[idx, 0], kde=True)
        axes[idx, 0].set_title(f'{col} (변환 전)')
        
        # 변환 후 분포
        sns.histplot(data=df_transformed, x=col, ax=axes[idx, 1], kde=True)
        axes[idx, 1].set_title(f'{col} (변환 후)')
    
    plt.tight_layout()
    plt.show()
    
    return df_transformed