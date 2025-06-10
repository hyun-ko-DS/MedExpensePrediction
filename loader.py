import os
import re
import pandas as pd

#############################################
# 1. 상수 정의
#############################################

dataset_path = '../data/medical'
codes = ['in', 'ind', 'cd', 'appen', 'hh', 'er', 'ou']
id_columns = ['HHIDWON', 'PIDWON']

#############################################
# 2. 데이터셋 경로 및 파일 관리 함수들
#############################################

def set_dataset_path() -> list[str]:
    """
    데이터셋 폴더 경로를 설정하고 SAS 파일 목록을 반환하는 함수
    
    Returns:
    --------
    list[str]
        SAS 파일 목록
    """
    # Dataset 폴더 내 파일 확인
    files = os.listdir(dataset_path)
    
    # SAS 데이터셋 파일만 필터링
    sas_files = [f for f in files if f.endswith('.sas7bdat')]
    
    return sas_files

def join_sas_by_code(files: list[str]) -> dict[str, list[str]]:
    """
    SAS 파일들을 DB 코드별로 분류하여 딕셔너리로 반환하는 함수
    
    Parameters:
    -----------
    files : list[str]
        SAS 파일 목록
    
    Returns:
    --------
    dict[str, list[str]]
        DB 코드별 파일 목록을 담은 딕셔너리
    """
    # 각 코드를 딕셔너리의 키로 배정
    data_dict = {code: [] for code in codes}
    
    # 전체 sas 파일들을 순회
    for file in files:
        # 정해진 파일 형식에서 연도와 DB 코드 추출
        pattern_match = re.match(r"t(\d{2})([a-zA-Z]+)\.sas7bdat", file)
        
        if pattern_match:
            year = pattern_match.group(1)
            code = pattern_match.group(2)
            
            # 각 DB 코드에 해당하는 파일명을 딕셔너리의 value로 지정
            data_dict[code].append(file)
    
    return data_dict

#############################################
# 3. 데이터 로딩 함수들
#############################################

def concat_by_code(file_dict: dict[str, list[str]], code: str) -> pd.DataFrame:
    """
    특정 DB 코드에 해당하는 모든 SAS 파일을 하나의 데이터프레임으로 결합하는 함수
    
    Parameters:
    -----------
    file_dict : dict[str, list[str]]
        DB 코드별 파일 목록을 담은 딕셔너리
    code : str
        처리할 DB 코드
    
    Returns:
    --------
    pandas.DataFrame
        결합된 데이터프레임
    """
    code_df = pd.DataFrame()
    files = file_dict[code]
    
    for file in files:
        # 정규표현식으로 연도 추출 (예: t14in.sas7bdat → 14 → 2014)
        match = re.search(r"t(\d{2})[a-zA-Z]+\.sas7bdat", file)
        if match:
            year = int("20" + match.group(1))  # "14" → "2014"
        else:
            print(f"⚠️ 연도 추출 실패: {file}")
            continue  # 연도 추출 실패 시 스킵
        
        # SAS 파일 읽기 및 연도 컬럼 추가
        curr_df = pd.read_sas(os.path.join(dataset_path, file))
        curr_df["YEAR"] = year
        
        # 데이터프레임 결합
        code_df = pd.concat([code_df, curr_df], axis=0, ignore_index=True)
    
    print(f"{code}코드 로드 완료!", "\n")
    
    return code_df
