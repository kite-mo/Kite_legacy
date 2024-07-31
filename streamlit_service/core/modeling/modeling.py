from typing import Any, Dict, List, Union

import pandas as pd

from utils.check_assertion import check_all_columns_in_df_list
from exceptions.exception import ModelingException
from modeling.functions import remove_outlier_dfs, get_model_dict


def modeling(df_list: List[pd.DataFrame], hm_column_list: List[str]) -> Dict[str, Any]:
    """
        1. 길이가 지나치게 짧거나 긴 DataFrame 제거
        2. 각 column에 정해진 rule 적용
        3. 각 column 별로 outlier 제거
        4. 에러 허용 margin 적용
        5. 화면 띄우기 용 정보 구함
        6. 대표 파형 생성
        7. scoring을 위한 거리 분포 계산
        8. model 정보 저장

    Args:
        df_list (List[pd.DataFrame]): model을 만들기 위한 DataFrame 리스트
        hm_column_list (List[str]): model을 만들 column 리스트
        outlier_percentile (float, optional): model을 만들 때, 버리는 데이터의 비율. Defaults to 0.25.

    Returns:
        Dict[str, Any]: model 정보
    """
    # 모든 DataFrame에 hm_column_list가 존재하는지 확인
    check_all_columns_in_df_list(
        df_list=df_list, columns_list=hm_column_list, exception=ModelingException
    )

    # Remove outlier df in df list
    df_list = remove_outlier_dfs(df_list=df_list)

    # Get model for each column
    model_dict: Dict[str, Any] = get_model_dict(
        df_list=df_list, hm_column_list=hm_column_list
    )

    return model_dict
