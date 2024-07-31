from typing import Any, List, Dict, Union, Tuple

from copy import deepcopy

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from exceptions.exception import PredictionException

from utils.utils import get_distance_bw_ts, get_path_bw_ts
from utils.check_assertion import check_all_values_not_None_in_dict
from utils.variable import scoring_mode_list
from rule.rule import (
    apply_rule_to_df_list,
    predict_by_rule,
)


def get_model_dict(
    model: Dict[str, Dict[str, Any]], column_list: List[str]
) -> Dict[str, Any]:
    """
        model로부터 각 scoring 방식에 맞게 필요한 정보를 뽑아 dictionary로 변환하여 return

    Args:
        model (Dict[str, Dict[str, Any]]): _description_
        column_list (List[str]): scoring할 column 리스트

    Raises:
        PredictionException: column이 hm_column_list에는 있으나 model에 없는 경우
        PredictionException: 구현되지 않은 방식으로 점수를 구하라고 함. (variable.py의 scoring_mode_list 참고)

    Returns:
        Dict[str, Any]: model로부터 각 column별로 필요한 정보만 뽑아 변환된 dictionary
    """
    model_dict: Dict[str, Any] = {column: None for column in column_list}

    for column in model_dict.keys():
        try:
            model_dict.update(
                {
                    column: {
                        "centerDict": model.get("centerDict").get(column),
                        "standardDistUpper": model.get("standardDist")
                        .get(column)
                        .get("upper"),
                        "standardDistLower": model.get("standardDist")
                        .get(column)
                        .get("lower"),
                    }
                }
            )
        except:
            message: str = f"{column} is in 'hm_column_list',"
            message == " but it is not in 'distanceDistribution' from model."
            raise PredictionException(message)

    return model_dict


def apply_rule(
    df: pd.DataFrame,
    column_list: List[str],
) -> Dict[str, np.ndarray]:
    """
        각 DataFrame에 주어진 rule을 적용한 후, 그 결과를 column별로 저장

    Args:
        df (pd.DataFrame): DataFrame
        column_list (List[str]): 전체 column 리스트

    Returns:
        Dict[str, np.ndarray]: 각 column 별 rule이 적용된 값
    """
    # 각 column 별로 DataFrame에 rule 적용 후, column 별로 dictionary에 저장하여 반환
    _, modified_values_list_dict = apply_rule_to_df_list(
        df_list=[df], hm_column_list=column_list, exception=PredictionException
    )

    # 위 함수는 df_list에 대한 함수이므로 각 column 별로 길이가 1인 List가 생성되었으므로 List의 element 하나만 가지고 나옴
    modified_values_dict: Dict[str, np.ndarray] = {
        column: values_list[0]
        for column, values_list in modified_values_list_dict.items()
    }

    # 모든 column에 해당하는 값이 None이 아님을 확인
    check_all_values_not_None_in_dict(
        dict=modified_values_dict, exception=PredictionException
    )

    return modified_values_dict


# ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ#
#############################             d_ratio ver 함수           ################################
# ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ#


def __get_target_dist_ratio_df(
    center_values: np.array,
    upper_values: np.array,
    lower_values: np.array,
    target_values: np.array,
    dtw_path: List[Tuple[int, int]],
) -> pd.DataFrame:
    precision = np.finfo(np.float32).eps
    target_dist_ratio_list = []

    for center_idx, target_idx in dtw_path:
        center_value = center_values[center_idx]
        target_value = target_values[target_idx]
        center_target_dist = center_value - target_value

        # dist_ratio = target_dist_array/(sensor_standard_dist + precision)
        ## 데이터 양수/음수 차이에 따른 upper/lower_standard_dist 적용
        if center_target_dist < 0:
            dist_ratio = np.abs(
                center_target_dist / (upper_values[center_idx] + precision)
            )
        else:
            dist_ratio = center_target_dist / (lower_values[center_idx] + precision)

        target_dist_ratio_list.append([center_idx, target_idx, dist_ratio])

    target_dist_ratio_df = pd.DataFrame(
        target_dist_ratio_list, columns=["center_idx", "target_idx", "dist_ratio"]
    )

    return target_dist_ratio_df


def moving_average_window_size(one_d_array: np.array, winodw_size: int = 3) -> np.array:
    moving_average_array = np.average(
        sliding_window_view(one_d_array, window_shape=winodw_size), axis=1
    )
    add_len = len(one_d_array) - len(moving_average_array)
    for idx in range(0, add_len):
        moving_average_array = np.append(moving_average_array, moving_average_array[-1])

    return moving_average_array


def __get_dist_ratio_by_target_idx(
    target_dist_ratio_df: pd.DataFrame, smoothing: bool = True
) -> np.array:
    grouped_ratio_df = target_dist_ratio_df.groupby(["target_idx"])
    mean_dist_ratio_by_target_idx = grouped_ratio_df.mean()["dist_ratio"].values

    if smoothing == True:
        dist_ratio_by_target_idx = moving_average_window_size(
            mean_dist_ratio_by_target_idx
        )
    else:
        dist_ratio_by_target_idx = mean_dist_ratio_by_target_idx

    return dist_ratio_by_target_idx


def __get_target_plotting_list(
    center_values: np.array,
    target_values: np.array,
    dtw_path: List[Tuple[int, int]],
) -> List[float]:
    target_plotting_dict = {}

    # center index 와 매칭되는 target value 를 채울 Dict(List) 생성
    for idx_key in range(0, len(center_values)):
        target_plotting_dict[idx_key] = []

    # center index 와 매칭되는 target value 를 채우기
    for center_idx, target_idx in dtw_path:
        target_value = target_values[target_idx]
        target_plotting_dict[center_idx].append(target_value)

    # center 와 길이가 다를 경우, target values 들을 조절하여 동일한 길이로 그리기 위함
    ## 현재는 중앙값으로 처리
    target_plotting_list = []
    for idx, target_values in target_plotting_dict.items():
        target_value = np.median(target_values)
        target_plotting_list.append(target_value)

    return target_plotting_list


####################################################################################################
####################################################################################################
####################################################################################################


def get_pred_dict(
    model_dict: Dict[str, np.ndarray], values_dict: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
        model의 prediction 결과 구하기

    Args:
        model_dict (Dict[str, np.ndarray]): model로부터 각 column별로 필요한 정보만 뽑아 변환된 dictionary
        values_dict (Dict[str, np.ndarray]): 각 column 별 rule이 적용된 값

    Raises:
        PredictionException: 구현되지 않은 방식으로 점수를 구하라고 함. (variable.py의 scoring_mode_list 참고)

    Returns:
        Dict[str, Any]: 각 column별 model의 prediction 결과
    """

    pred_dict: Dict[str, Any] = {column: None for column in model_dict.keys()}

    for column in pred_dict.keys():
        # 각 column의 모델 불러오기
        model: Any = model_dict.get(column)
        # 각 column의 값 불러오기
        column_values: np.ndarray = values_dict.get(column)
        pred: Any = None

        model_ = deepcopy(model)
        center = model_.get("centerDict")
        upper = model_.get("standardDistUpper")
        lower = model_.get("standardDistLower")

        # get dtw_path between center and target
        dtw_path = get_path_bw_ts(
            center=center, target=column_values, exception=PredictionException
        )

        # get dist_ratio for scoring
        target_dist_ratio_df = __get_target_dist_ratio_df(
            center_values=center,
            target_values=column_values,
            upper_values=upper,
            lower_values=lower,
            dtw_path=dtw_path,
        )
        dist_ratio_by_target_idx = __get_dist_ratio_by_target_idx(
            target_dist_ratio_df=target_dist_ratio_df
        )

        # get target plot list for solution frontend
        target_plotting_list = __get_target_plotting_list(
            center_values=center, target_values=column_values, dtw_path=dtw_path
        )

        pred_by_rule_dict = {}
        pred_by_rule_dict.update(
            {"center": center, "target": target_plotting_list, "column": column}
        )
        pred_by_rule_dict.update({"dist_ratio": dist_ratio_by_target_idx})
        pred = pred_by_rule_dict

        pred_dict.update({column: pred})

    check_all_values_not_None_in_dict(dict=pred_dict, exception=PredictionException)

    return pred_dict


def make_result(pred_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
        원하는 방식으로 결과 저장 방식 수정

    Args:
        pred_dict (Dict[str, Any]): 각 column별 model의 prediction 결과

    Raises:
        PredictionException: 구현되지 않은 방식으로 점수를 구하라고 함. (variable.py의 scoring_mode_list 참고)

    Returns:
        Dict[str, Dict[str, Any]]: 원하는 방식으로 수정된 결과
    """

    result: Dict[str, Dict[str, Union[float, np.ndarray]]] = {
        "dist_ratio": pred_dict,
    }

    return result
