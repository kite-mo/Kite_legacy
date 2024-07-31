from typing import Any, Union, List, Dict

from copy import deepcopy

import numpy as np

from exceptions.exception import ScoringException

from .variable import (
    precision,
    scoring_mode_list,
    valid_dist_method_list,
    rules,
    dist_metric,
    dtw_metric_params,
)

from tslearn.metrics import dtw_path


def __get_score_d_max(
    dist: float,
    threshold: float,
    std: float,
    alpha: float,
    threshold_score: float = 0.8,
    shape: float = 0.5,
) -> float:
    """
        d_max를 활용한 scoring 방법
        1. d_max에 해당하는 dist는 80점 (threshold_score)
            1-1. dist가 d_max보다 작다면 80 ~ 100점
            1-2. dist가 d_max보다 크다면 0 ~ 80점
        * 계산 식 상 std가 소거되어 그냥 dist와 threshold 활용
        * std가 너무 작은 경우, 계산상 overflow가 발생할 수 있기 때문에 d_std 제외

    Args:
        dist (float): 평가 대상
        threshold (float): 80점(threshold_score)를 받는 기준
        std: normalized를 위한 표준편차
        alpha (float): 보정치
        threshold_score (float, optional): 기준 점수. Defaults to 0.8.
        shape (float, optional): 80점 이하에서 점수가 떨어지는 속도. Defaults to 0.5.

    Returns:
        float: 평가 대상의 점수
    """

    b = -6
    a = np.log((1 + np.exp(b)) / threshold_score - 1) - b

    x = ((dist / alpha) / (threshold + precision)) ** shape
    ax_b = a * x + b
    # overflow 방지
    if ax_b > 706:
        ax_b = 706

    score = (1 + np.exp(b)) / (1 + np.exp(ax_b))

    return score


# dist ratio 를 score 로 변환해주는 함수
def __get_score_d_ratio(
    score_ratio: np.ndarray,
) -> np.ndarray:
    def _convert_ndist2score(ndist, d_max, left_end, shape=0.75):
        # ndist2score func by heelang
        # x >= 0
        # d_max : max among 24 distances
        # shape : must be postivie. larger, steeper.

        precision = np.finfo(np.float32).eps

        ndist = ndist.astype("float64")
        d_max = d_max.astype("float64")
        left_end = np.array([left_end]).astype("float64")

        score_dmax = 0.80

        score = np.zeros_like(ndist)
        cal_idx = ndist >= 0

        # normalize 된 거리를 로지스틱 함수를 통해 점수로 변환
        b = -6
        ax = (np.log((1 + np.exp(b)) / (score_dmax / 0.95) - 1) - b) * (
            (ndist[cal_idx] / (d_max[0] + precision)) ** shape
        )
        ax_b = np.where(
            ax + b > 700, 700, ax + b
        )  # to avoid overflow warning(np.exp(X) == inf, where X is larger than 706)

        # ndist가 음수인 경우는 center와 아주 가까우니까 이후에 처리, 양수인 경우는 점수 계산 해주기
        score[cal_idx] = (1 + np.exp(b)) / (1 + np.exp(ax_b)) * 0.95

        # left end 일때가 100점
        cal_idx2 = ndist < (left_end + precision)
        random_add = np.random.uniform(0, 0.02, size=score.shape)
        score[cal_idx2] = 0.98 + random_add[cal_idx2]

        # ndist가 아주 가까운 경우는 95점부터 leftend까지 linear하게 처리해줌
        cal_idx3 = ~(cal_idx | cal_idx2)
        score[cal_idx3] = (1 - 0.95) * (
            ndist[cal_idx3] / (left_end[0] - precision)
        ) + 0.95

        cal_idx4 = np.isnan(score)
        score[cal_idx4] = 0.98 + random_add[cal_idx4]

        # 안흔들리는 segment 100점 처리
        one_idx = (ndist <= precision) & (d_max <= precision)
        score[one_idx] = 1

        score = score.astype("float32")
        return score

    d_max = np.array([1.0])
    point_score = _convert_ndist2score(score_ratio, d_max, 0)

    return point_score.astype("float32")


def __get_score(
    pred: Any,
) -> float:
    """
        pred의 점수를 model을 기준으로 구함

    Args:
        pred (Any): 평가 대상

    Raises:
        ScoringException: 구현되지 않은 방식으로 점수를 구하라고 함. (variable.py의 scoring_mode_list 참고)

    Returns:
        float: 평가 대상의 점수
    """

    dist_ratio = deepcopy(pred.get("dist_ratio"))
    score = __get_score_d_ratio(score_ratio=dist_ratio)

    return score


def get_score_dict(pred_dict: Dict[str, float]) -> Dict[str, float]:
    """
        각 column의 점수를 구함

    Args:
        pred_dict (Dict[str, float]): prediction으로부터 각 column별로 필요한 정보만 뽑아 변환된 dictionary

    Returns:
        Dict[str, float]: 각 column별 점수
    """

    return {
        column: __get_score(
            pred=pred_dict.get(column),
        )
        for column in pred_dict.keys()
    }


def get_model_dict(
    model: Dict[str, Dict[str, Any]], column_list: List[str]
) -> Dict[str, Any]:
    """
        model로부터 각 scoring 방식에 맞게 필요한 정보를 뽑아 dictionary로 변환하여 return

    Args:
        model (Dict[str, Dict[str, Any]]): model의 모든 정보
        column_list (List[str]): scoring할 column 리스트
        mode (str, optional): score 구하는 방식. Defaults to 'd_max'.

    Raises:
        ScoringException: column이 hm_column_list에는 있으나 model의 dist_distribution_dict에 없는 경우
        ScoringException: 구현되지 않은 방식으로 점수를 구하라고 함. (variable.py의 scoring_mode_list 참고)

    Returns:
        Dict[str, Any]: model로부터 각 column별로 필요한 정보만 뽑아 변환된 dictionary
    """
    model_dict = {column: None for column in column_list}

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
            raise ScoringException(
                f"{column} is in 'hm_column_list', but it is not in 'standardDist' from model."
            )

    return model_dict


def get_pred_dict(
    prediction: Dict[str, Dict[str, Any]], column_list: List[str]
) -> Dict[str, Any]:
    """
        prediction으로부터 각 scoring 방식에 맞게 필요한 정보를 뽑아 dictionary로 변환하여 return

    Args:
        prediction (Dict[str, Dict[str, Any]]): prediction의 모든 정보
        column_list (List[str]): scoring할 column 리스트

    Raises:
        ScoringException: column이 hm_column_list에는 있으나 prediction의 dist_dict에 없는 경우
        ScoringException: 구현되지 않은 방식으로 점수를 구하라고 함. (variable.py의 scoring_mode_list 참고)

    Returns:
        Dict[str, Any]: prediction으로부터 각 column별로 필요한 정보만 뽑아 변환된 dictionary
    """
    pred_dict = {column: None for column in column_list}

    for column in pred_dict.keys():
        try:
            pred_dict.update(
                {
                    column: {
                        "dist_ratio": prediction.get("dist_ratio")
                        .get(column)
                        .get("dist_ratio"),
                    }
                }
            )
        except:
            raise ScoringException(
                f"{column} is in 'hm_column_list', but it is not in 'dist_ratio' from prediction."
            )

    return pred_dict
