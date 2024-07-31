from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import copy

from exceptions.exception import ModelingException

from utils.utils import (
    get_distance_bw_ts,
    get_path_bw_ts,
    get_ts_barycenter,
    get_representative_value,
)
from utils.check_assertion import check_all_values_not_None_in_dict
from utils.variable import scoring_mode_list, precision
from rule.rule import apply_rule_to_df_list, apply_rule_for_model_dicts


####################################################################################################
####################################################################################################
######################        df_list의 Outlier 제거 관련 함수       ###############################
def remove_outlier_dfs(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
        outlier들을 제거한 뒤 'df_list' 반환

    Args:
        df_list (List[pd.DataFrame]): 전체 DataFrame 리스트

    Raises:
        ModelingException: 모든 DataFrame이 제거됨

    Returns:
        List[pd.DataFrame]: outlier들이 제거된 DataFrame 리스트
    """

    outlier_index: List[int] = __get_outlier_index(df_list=df_list)

    if len(outlier_index) == len(df_list):
        message: str = "All of DataFrames in 'df_list' are outliers."
        raise ModelingException(message)

    return [df for df_idx, df in enumerate(df_list) if df_idx not in outlier_index]


def __get_outlier_index(df_list: List[pd.DataFrame]) -> List[int]:
    """
        'df_list'에서 outlier에 해당하는 index 찾기
        TODO: 더 많은 outlier 방법론 구현 (if needed)

    Args:
        df_list (List[pd.DataFrame]): 전체 DataFrame 리스트

    Returns:
        List[int]: outlier의 'df_list'에서의 index
    """

    def _find_by_length(temp_list: List[int]) -> np.ndarray:
        q1, q3 = np.percentile(temp_list, [25, 75])
        iqr: float = q3 - q1
        iqr = 1 if iqr == 0 else iqr
        lb: float = q1 - (iqr * 3)
        ub: float = q3 + (iqr * 3)
        return np.where((temp_list > ub) | (temp_list < lb))[0]

    # find outlier by reference wafer lengths distribution
    df_length_list: List[int] = [len(df) for df in df_list]
    length_outlier_index: np.ndarray = _find_by_length(df_length_list)

    return length_outlier_index.tolist()


# ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ#
#############################             d_ratio ver 함수           ################################
# ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ#


def __get_reference_matching_dict(modified_values_list_dict: Dict):
    reference_matching_dict = {}  # reference 의 정보를 담은 dictionary
    rep_sensor_center_dict = {}  # reference 웨이퍼들의 center 정보를 담은 dictionary
    rep_sensor_center_idx_matching_dict = (
        {}
    )  # center 와 매칭되는 reference 웨이퍼들의 idx 를 담을 dictionary
    rep_sensor_seg_len_dict = {}  # wafer step 당 세그먼트 길이를 담을 dictionary

    for sensor, modified_values_list in modified_values_list_dict.items():
        # 센서의 DBA 뽑기
        sensor_barycenter = get_ts_barycenter(
            values_list=modified_values_list, exception=ModelingException
        )

        # 센터와 reference 후보 웨이퍼 간 DTW 거리 구하기
        dtw_dist_from_center = [
            get_distance_bw_ts(
                x=sensor_barycenter, y=modified_value, exception=ModelingException
            )
            for modified_value in modified_values_list
        ]
        # 센터와 가장 가까운 웨이퍼 선정
        nearest_values_sensor_with_center = modified_values_list[
            np.argmin(dtw_dist_from_center)
        ]

        # TODO : 레시피 스탭 당 데이터 길이 추가
        # seg_len_dict = ~~~ 추가 필요

        # 센터와 매칭되는 reference 웨이퍼들의 idx 를 담을 공간 만들기
        rep_sensor_center_idx_matching_dict[sensor] = {}
        for center_idx in range(0, len(nearest_values_sensor_with_center)):
            rep_sensor_center_idx_matching_dict[sensor][center_idx] = []

        # 센터의 각 인덱스와 매칭되는 reference 웨이퍼들의 센서의 실제값 담기
        for modified_values in modified_values_list:
            dtw_path = get_path_bw_ts(
                center=nearest_values_sensor_with_center,
                target=modified_values,
                exception=ModelingException,
            )

            for center_idx, target_idx in dtw_path:
                modified_value = modified_values[target_idx]
                rep_sensor_center_idx_matching_dict[sensor][center_idx].append(
                    modified_value
                )

        # rep_sensor_seg_len_dict[sensor] = seg_len_dict
        rep_sensor_center_dict[sensor] = nearest_values_sensor_with_center

    reference_matching_dict = {
        "center_dict": rep_sensor_center_dict,
        "center_matching_dict": rep_sensor_center_idx_matching_dict,
    }  # 'rep_seg_len_dict' : rep_sensor_seg_len_dict

    return reference_matching_dict


def __get_sensor_points_std_dict(reference_matching_dict, hm_column_list):
    sensor_points_std_dict = {}

    for sensor in hm_column_list:
        sensor_points_std_dict[sensor] = {}
        sensor_points_std_dict[sensor]["upper"] = []
        sensor_points_std_dict[sensor]["lower"] = []

        sensor_center = reference_matching_dict["center_dict"][sensor]
        sensor_matching_dict = reference_matching_dict["center_matching_dict"][sensor]

        for center_idx, matched_values in sensor_matching_dict.items():
            center_value = sensor_center[center_idx]

            # split by bigger than 0 or smaller than 0
            dist_center_and_targets = center_value - np.array(matched_values)
            positive_diff = dist_center_and_targets[
                np.where(dist_center_and_targets >= 0)[0]
            ]
            negative_diff = dist_center_and_targets[
                np.where(dist_center_and_targets <= 0)[0]
            ]
            # TODO : FIX : 표본 개수가 작은 경우, 합리적인 std 값이 만들어지지 않음
            positive_points_std, negative_points_std = np.std(positive_diff), np.std(
                negative_diff
            )

            # upper : target 이 reference 보다 값이 큰 경우
            # lower : target 이 reference 보다 값이 작은 경우
            sensor_points_std_dict[sensor]["upper"].append(negative_points_std)
            sensor_points_std_dict[sensor]["lower"].append(positive_points_std)

    return sensor_points_std_dict


def __get_standard_scoring_dict(
    reference_matching_dict, sensor_points_std_dict, hm_column_list, sigma_range=3
):
    standard_scoring_dict = {}

    # make threshold sigma info
    # upper : target 이 reference 보다 값이 큰 경우
    # lower : target 이 reference 보다 값이 작은 경우
    sensor_points_upper_sigma_dict = {}
    sensor_points_lower_sigma_dict = {}
    sensor_scoring_standard_dict = {}

    for sensor in hm_column_list:
        sensor_scoring_standard_dict[sensor] = {}
        sensor_center = reference_matching_dict["center_dict"][sensor]

        # 그림 그리기 위한 upper, lower range 저장 쁠마 3시그마 기준
        upper_std, lower_std = (
            sensor_points_std_dict[sensor]["upper"],
            sensor_points_std_dict[sensor]["lower"],
        )
        sensor_points_upper_sigma_dict[sensor] = list(
            sensor_center + (sigma_range * np.array(upper_std))
        )
        sensor_points_lower_sigma_dict[sensor] = list(
            sensor_center - (sigma_range * np.array(lower_std))
        )

        # 점수를 산정하기 위한 포인트 별 기준 거리
        upper_sigma, lower_sigma = (
            sensor_points_upper_sigma_dict[sensor],
            sensor_points_lower_sigma_dict[sensor],
        )
        standard_upper_dist = upper_sigma - sensor_center + precision
        standard_lower_dist = sensor_center - lower_sigma + precision

        sensor_scoring_standard_dict[sensor]["upper"] = standard_upper_dist
        sensor_scoring_standard_dict[sensor]["lower"] = standard_lower_dist

    standard_scoring_dict["center_dict"] = copy.deepcopy(
        reference_matching_dict["center_dict"]
    )
    # standard_scoring_dict['seg_len_dict'] = copy.deepcopy(reference_matching_dict['seg_len_dict'])
    standard_scoring_dict["std_dict"] = sensor_points_std_dict
    standard_scoring_dict["upper_sigma"] = sensor_points_upper_sigma_dict
    standard_scoring_dict["lower_sigma"] = sensor_points_lower_sigma_dict
    standard_scoring_dict["standard_dist"] = sensor_scoring_standard_dict

    return standard_scoring_dict


####################################################################################################
####################################################################################################
####################################################################################################


def get_model_dict(
    df_list: List[pd.DataFrame], hm_column_list: List[str]
) -> Dict[str, Any]:
    model_dict: Dict[str, Any] = {}

    # 1. Apply rules to each column & Change to dictionary (key: column, value: List of numpy array)
    values_list_dict, modified_values_list_dict = apply_rule_to_df_list(
        df_list=df_list, hm_column_list=hm_column_list, exception=ModelingException
    )

    # 2. Reference 웨이퍼 들의 센서 당 대표 센터 연산
    reference_matching_dict = __get_reference_matching_dict(
        modified_values_list_dict=modified_values_list_dict
    )

    # 3. center 포인트 별 매칭되는 reference wafer 센서값들의 표준편차 - 음수, 양수 차이에 따른
    sensor_points_std_dict = __get_sensor_points_std_dict(
        reference_matching_dict=reference_matching_dict, hm_column_list=hm_column_list
    )

    # 4. 최종 쁠마 N 시그마에 대한 그림 그리기 위한 정보와 점수 계산하기 위한 정보
    standard_scoring_dict = __get_standard_scoring_dict(
        reference_matching_dict=reference_matching_dict,
        sensor_points_std_dict=sensor_points_std_dict,
        hm_column_list=hm_column_list,
    )
    center_dict = standard_scoring_dict["center_dict"]
    rep_max_dict = standard_scoring_dict["upper_sigma"]
    rep_min_dict = standard_scoring_dict["lower_sigma"]
    standard_dist_dict = standard_scoring_dict["standard_dist"]

    model_dict["centerDict"] = center_dict
    model_dict["repDict"] = center_dict
    model_dict["repMaxDict"] = rep_max_dict
    model_dict["repMinDict"] = rep_min_dict
    model_dict["standardDist"] = standard_dist_dict

    check_all_values_not_None_in_dict(dict=model_dict, exception=ModelingException)

    return model_dict
