from typing import Any, List, Dict, Union

import numpy as np
import pandas as pd

from prediction.functions import get_model_dict, apply_rule, get_pred_dict, make_result


def prediction(
    df: pd.DataFrame, model: Dict[str, Any], hm_column_list: List[str]
) -> Dict[str, Any]:
    """
        1. 각 column에 대한 model 불러오기
        2. DataFrame의 각 column에 정해진 rule 적용
        3. 각 column에 대한 pred 예측
        4. 결과 저장

    Args:
        df (pd.DataFrame): DataFrame
        model (Dict[str, Any]): modeling에서 구한 model
        hm_column_list (List[str]): scoring을 낼 column 리스트

    Returns:
        Dict[str, Any]: 결과 정보
    """

    # 1. Get model for each column
    model_dict: Dict[str, Any] = get_model_dict(model=model, column_list=hm_column_list)

    # 2. Apply Rules to each column & Change into dictionary (key: column, value: numpy array)
    df_dict: Dict[str, np.ndarray] = apply_rule(df=df, column_list=hm_column_list)

    # print("AI PREDICTION START!")

    # 3. Get prediction for each column from each model
    pred_dict: Dict[str, Any] = get_pred_dict(
        model_dict=model_dict, values_dict=df_dict
    )

    # print("AI PREDICTION END!")

    # 4. Make result
    result: Dict[str, Dict[str, Any]] = make_result(pred_dict=pred_dict)

    return result
