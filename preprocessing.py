import pandas as pd
import numpy as np
import duckdb
from scipy import stats

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
        
        # 연도 차이 계산
        df[col] = np.where(
            target_col.isnull(),
            -1,
            year_col - target_col#.fillna(0)
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
            # PIDWON으로 그룹화하여 이전 값 + 1로 채우기
            temp_forward = clean_df.groupby('PIDWON')[col].fillna(method='ffill') + 1
            # 이전 값이 없는 경우에만 다음 값 - 1로 채우기
            clean_df[col] = temp_forward.fillna(clean_df.groupby('PIDWON')[col].fillna(method='bfill') - 1)
            # 그럼에도 값이 없는 경우에는 -1로 대체
            clean_df[col] = clean_df[col].fillna(-1)
            
        elif col in numeric_continuous_cols and col not in [user_cols, incremental_cols, constant_cols]:  # 수치형 - 연속형 변수는 0으로 대체 (주로 의료 비용 관련 변수)
            # PIDWON으로 그룹화하여 각 가구원별 중앙값으로 채우기
            # 중앙값을 계산할 때 정수형으로 변환
            # print(col)
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


# 환자가, 특정 연도 데이터만 있고, 나머지는 없는 경우에도
# 각 연도별로 데이터 행을 생성해줘야 함.
# 없는 연도에 대해서 행을 추가하고, 그 값을 0으로 업데이트하는 함수.
# def fill_na_years(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     각 환자(PIDWON)별로 2014~2018년도의 모든 행이 존재하도록 채웁니다.
#     없는 연도의 행은 0으로 채워집니다.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         입력 데이터프레임
    
#     Returns:
#     --------
#     pandas.DataFrame
#         모든 연도가 채워진 데이터프레임
#     """
#     # 고유한 (HHIDWON, PIDWON) 쌍 추출
#     unique_pairs = df[['HHIDWON', 'PIDWON']].drop_duplicates()
    
#     # 모든 연도 리스트
#     all_years = [2014, 2015, 2016, 2017, 2018]
    
#     # 모든 조합 생성
#     full_combinations = []
#     for _, row in unique_pairs.iterrows():
#         for year in all_years:
#             full_combinations.append({
#                 'HHIDWON': row['HHIDWON'],
#                 'PIDWON': row['PIDWON'],
#                 'YEAR': year
#             })
    
#     # 새로운 데이터프레임 생성
#     full_df = pd.DataFrame(full_combinations)
    
#     # 원본 데이터프레임과 병합
#     result_df = pd.merge(
#         full_df,
#         df,
#         on=['HHIDWON', 'PIDWON', 'YEAR'],
#         how='left'
#     )
    
#     # 숫자형 컬럼의 NaN 값을 0으로 채움
#     numeric_columns = result_df.select_dtypes(include=['int64', 'float64']).columns
#     result_df[numeric_columns] = result_df[numeric_columns].fillna(0)
    
#     # 범주형 컬럼의 NaN 값을 0으로 채움
#     categorical_columns = result_df.select_dtypes(include=['category']).columns
#     result_df[categorical_columns] = result_df[categorical_columns].fillna(-1)
    
#     # 불리언 컬럼의 NaN 값을 False로 채움
#     boolean_columns = result_df.select_dtypes(include=['bool']).columns
#     result_df[boolean_columns] = result_df[boolean_columns].fillna(False)
    
#     return result_df

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

# def handle_missing_values(df: pd.DataFrame, missing_cols: list) -> pd.DataFrame:
#     """
#     컬럼 유형별로 결측치를 적절한 값으로 대체합니다.
    
#     1. er, in이 포함된 컬럼: 결측치를 -1로 대체
#     2. ou, smoking, drinking, workout, mobility, physical, depression, unavailability가 포함된 컬럼:
#        - 같은 PIDWON의 이전 연도 값으로 대체 (ffill)
#        - 2014년인 경우 이후 연도 값으로 대체 (bfill)
#        - 여전히 null 값이 남아있는 경우 -1로 대체
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         입력 데이터프레임
#     missing_cols : list
#         결측치가 있는 컬럼명 리스트
    
#     Returns:
#     --------
#     pandas.DataFrame
#         결측치가 처리된 데이터프레임
#     """
#     # 1. er, in이 포함된 컬럼: 결측치를 -1로 대체
#     er_in_cols = [col for col in missing_cols if any(x in col.lower() for x in ['er', 'in'])]
#     df[er_in_cols] = df[er_in_cols].fillna(-1)
    
#     # 2. ou, smoking, drinking, workout, mobility, physical, depression, unavailability가 포함된 컬럼
#     target_cols = [col for col in missing_cols if any(x in col.lower() for x in [
#         'ou', 'smoking', 'drinking', 'workout', 'mobility', 
#         'physical', 'depression', 'unavailability'
#     ])]
    
#     # 연도별로 정렬된 데이터프레임 생성
#     df_sorted = df.sort_values(['PIDWON', 'YEAR'])
    
#     # 각 PIDWON별로 이전 연도 값으로 채우기 (ffill)
#     df_sorted[target_cols] = df_sorted.groupby('PIDWON')[target_cols].fillna(method='ffill')
    
#     # 2014년 데이터에 대해 이후 연도 값으로 채우기 (bfill)
#     df_sorted.loc[df_sorted['YEAR'] == 2014, target_cols] = df_sorted.loc[df_sorted['YEAR'] == 2014, target_cols].fillna(method='bfill')
    
#     # 여전히 null 값이 남아있는 경우 -1로 대체
#     df_sorted[target_cols] = df_sorted[target_cols].fillna(-1)
    
#     # 원래 순서로 되돌리기
#     df = df_sorted.sort_index()
    
#     return df

# def separate_categorical_columns(df: pd.DataFrame) -> tuple:
#     """
#     범주형 변수들을 순서형과 명목형으로 구분합니다.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         입력 데이터프레임
    
#     Returns:
#     --------
#     tuple
#         (순서형 변수 리스트, 명목형 변수 리스트)
#     """
#     # ID 컬럼 제외
#     pure_columns = list(set(df.columns.tolist()) - set(id_columns))
    
#     # 범주형 변수 추출
#     categorical_cols = []
#     for col in pure_columns:
#         if df[col].dtype == 'object' or df[col].dtype == 'category' or df[col].nunique() < 20:
#             categorical_cols.append(col)
    
#     # 순서형 변수 키워드
#     ordinal_keywords = [
#         'level', 'grade', 'stage', 'degree', 'rank', 'score',
#         'income', 'education', 'age', 'year', 'month', 'day',
#         'duration', 'period', 'time', 'count', 'number'
#     ]
    
#     # 순서형과 명목형 변수 구분
#     ordinal_cols = []
#     nominal_cols = []
    
#     for col in categorical_cols:
#         # 컬럼명에 순서형 키워드가 포함되어 있는지 확인
#         is_ordinal = any(keyword in col.lower() for keyword in ordinal_keywords)
        
#         # 값의 분포 확인
#         value_counts = df[col].value_counts()
#         unique_values = df[col].nunique()
        
#         # 순서형 판단 기준:
#         # 1. 컬럼명에 순서형 키워드가 포함되어 있거나
#         # 2. 값이 숫자형이고 연속적인 분포를 보이거나
#         # 3. 값의 개수가 적고 순차적인 분포를 보이는 경우
#         if (is_ordinal or 
#             (pd.api.types.is_numeric_dtype(df[col]) and unique_values < 10) or
#             (unique_values < 10 and all(str(i) in value_counts.index for i in range(1, unique_values + 1)))):
#             ordinal_cols.append(col)
#         else:
#             nominal_cols.append(col)
    
#     return ordinal_cols, nominal_cols

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
    
#     # 1. 연속형 X와 연속형 Y: Pearson 상관분석
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
    
#     # 2. 순서형 X와 연속형 Y: Spearman 상관분석
#     for col in ordinal_cols:
#         corr, p_value = stats.spearmanr(df[col].dropna(), df[target_col].dropna())
#         results[col] = {
#             'type': 'ordinal',
#             'test': 'spearman',
#             'correlation': corr,
#             'p_value': p_value
#         }
    
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
    
#     return results