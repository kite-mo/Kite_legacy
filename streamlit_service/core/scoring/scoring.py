from typing import Any, List, Dict, Union
from .functions import get_score_dict, get_model_dict, get_pred_dict
import numpy as np


# TODO: Step 처리를 AI에서 해야한다고 함
def scoring(
    model: Dict[str, Any], prediction: Dict[str, Any], hm_column_list: List[str]
) -> Dict[str, Any]:
    """
        1. prediction의 결과와 modeling의 결과 불러오기
        2. score 구하기
        3. 결과 저장

    Args:
        model (Dict[str, Any]): modeling의 결과
        prediction (Dict[str, Any]): prediction의 결과
        hm_column_list (List[str]): score를 낼 column 리스트
        alpha_dict (Union[None, Dict[str, float]], optional): 각 column 별 alpha. Defaults to None.

    Returns:
        Dict[str, Any]: 각 column의 score 및 percent score
    """

    # 1. prediction의 결과와 modeling의 결과 불러오기
    pred_dict = get_pred_dict(prediction=prediction, column_list=hm_column_list)
    model_dict = get_model_dict(model=model, column_list=hm_column_list)

    # print("AI SCORING START!")

    # 2. score 구하기
    score_dict = get_score_dict(pred_dict=pred_dict)

    # print("AI SCORING END!")

    # 3. 결과 저장
    result = {
        "score": score_dict,
        "percentScore": {column: 100 * score for column, score in score_dict.items()},
    }

    sensor_score = {
        sensor: round(np.mean(score_array), 3)
        for sensor, score_array in result["percentScore"].items()
    }
    wafer_score = np.mean(
        [sensor_score for sensor, sensor_score in sensor_score.items()]
    )

    result["sensorScore"] = sensor_score
    result["waferScore"] = wafer_score

    return result
