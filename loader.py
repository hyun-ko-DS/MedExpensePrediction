import os
import re
import pandas as pd

dataset_path = "data"
codes = ['in', 'ind', 'cd', 'appen', 'hh', 'er', 'ou']
id_columns = ['HHIDWON', 'PIDWON']

def set_dataset_path() -> list:
    # Dataset 폴더 경로 설정
    dataset_path = 'data'

    # Dataset 폴더 내 파일 확인
    files = os.listdir(dataset_path)

    # 모든 파일 및 sas 데이터셋 확인
    all_files = files
    sas_files = [f for f in files if f.endswith('.sas7bdat')]

    print(f"모든 파일 목록: {all_files}")
    print(f"모든 파일 길이 수: {len(all_files)}")
    print(f"SAS 파일 목록: {sas_files}")
    print(f"SAS 파일 길이 수: {len(sas_files)}")

    return sas_files


# 각 DB 코드 별로 데이터셋을 딕셔너리화하여 묶음.   
def join_sas_by_code(files: list) -> dict:
  codes = ['in', 'ind', 'cd', 'appen', 'hh', 'er', 'ou'] # 전체 DB 코드
  data_dict = {code: [] for code in codes} # 각 코드를 딕셔너리의 키로 배정.

  # 전체 sas 파일들을 순회.
  for file in files:
    # 각 sas 파일 별 경로 지정
    file_path = os.path.join(dataset_path, file)

    # 정해진 파일 형식에서 연도와 DB 코드 추출
    pattern_match = re.match(r"t(\d{2})([a-zA-Z]+)\.sas7bdat", file)

    year = pattern_match.group(1)
    code = pattern_match.group(2)

    # 각 DB 코드에 해당하는 파일명을 딕셔너리의 value로 지정.
    data_dict[code].append(file)
  return data_dict


def concat_by_code(file_dict: dict, code: str) -> pd.DataFrame:
  code_df = pd.DataFrame()
  files = file_dict[code]
#   print(f"{code}코드 읽기 시작...")
  for file in files:
    # 정규표현식으로 연도 추출 (예: t14in.sas7bdat → 14 → 2014)
    match = re.search(r"t(\d{2})[a-zA-Z]+\.sas7bdat", file)
    if match:
        year = int("20" + match.group(1))  # "14" → "2014"
    else:
        print(f"⚠️ 연도 추출 실패: {file}")
        continue  # 연도 추출 실패 시 스킵

    curr_df = pd.read_sas(os.path.join(dataset_path, file))
    curr_df["YEAR"] = year
    # print(f"{file} 읽기 완료!")
    code_df = pd.concat([code_df, curr_df], axis = 0, ignore_index = True)
    print(f"{file} concat 완료!")
  print(f"{code}코드 로드 완료!", "\n")

  return code_df
