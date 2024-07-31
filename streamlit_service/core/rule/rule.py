from typing import Any, Dict, List, Tuple, Union

from copy import deepcopy

import sys

import numpy as np
import pandas as pd

from rule.columns import *
from rule.modify_df import *

from utils.check_assertion import *
from utils.utils import get_path_bw_ts
from utils.variable import rules

from exceptions.logic import LogicException


__all__ = ["apply_rule_to_df_list", "apply_rule_for_model_dicts"]

RULES_DICT: Dict = {}


def __get_rules_for_each_column(
    columns_list: List[str], exception: LogicException
) -> None:
    """
        RULES_DICT에 각 column별 적용할 rule 채워넣기

    Args:
        columns_list (List[str]): 모든 column 리스트

    Raises:
        exception: 해당 column이 두 개 이상의 sensor_list에 포함되어 있음
    """
    global RULES_DICT

    for column in columns_list:
        applied_rule: List[Dict[str, Any]] = []
        for rule in rules:
            sensor_list: List[str] = rule.get("sensor_list")
            if column in sensor_list:
                applied_rule.append(rule.get("how"))

        # 각 column이 하나의 sensor_list에만 속해있는지 확인
        check_one_rule_for_one_column(
            applied_rule=applied_rule, column=column, exception=exception
        )

        # RULES_DICT에 저장
        if len(applied_rule) == 1:
            rule_dict: Dict[str, Any] = applied_rule[0]
            RULES_DICT.update({column: rule_dict})
        else:
            RULES_DICT.update({column: {}})

    # 각 rule이 EHM에 맞게 설정되어있는지 확인
    check_rule_requirements(rule_dicts=RULES_DICT, exception=exception)


def apply_rule_to_df_list(
    df_list: List[pd.DataFrame],
    hm_column_list: List[str],
    exception: LogicException,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """
        각 column 별로 DataFrame에 rule 적용 후, column 별로 dictionary에 저장하여 반환
        # TODO: exclude, include 등 적용 (apply_{rule_name}_for_values 함수들에서)

    Args:
        df_list (List[pd.DataFrame]): DataFrame 리스트
        hm_column_list (List[str]): 모든 column 리스트

    Returns:
        Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]: (기존 데이터 dictionary,
            변경된 데이터 dictionary)
    """
    data = {column: [] for column in hm_column_list}
    modified_data = {column: [] for column in hm_column_list}

    # RULES_DICT에 값을 채워넣음
    __get_rules_for_each_column(columns_list=hm_column_list, exception=exception)

    # DataFrame 순서대로
    for df in df_list:
        # column 별로
        for column in hm_column_list:
            df_column_value = df[column].values
            modified_value = df_column_value.copy()

            # column 별로 dictionary에 저장
            data[column].append(df_column_value)
            modified_data[column].append(modified_value)

    check_all_values_not_None_in_dict(data, exception)
    check_all_values_not_None_in_dict(modified_data, exception)

    return data, modified_data


def __get_modified_value_for_center_and_rep(
    column: str,
    center_value: float,
    rep_value: float,
    rep_max_value: float,
    rep_min_value: float,
    rule_dict: Dict[str, Dict[str, Any]],
    exception: LogicException,
) -> Tuple[float, float, float, float]:
    modified_center_value = center_value
    modified_rep_value = rep_value
    modified_rep_max_value = rep_max_value
    modified_rep_min_value = rep_min_value

    for rule_name, rule_info in rule_dict.items():
        (
            modified_center_value,
            modified_rep_value,
            modified_rep_max_value,
            modified_rep_min_value,
        ) = getattr(sys.modules[__name__], f"apply_{rule_name}_for_center_and_rep")(
            {
                "column": column,
                "rule_info": rule_info,
                "modifiedCenterValue": modified_center_value,
                "modifiedRepValue": modified_rep_value,
                "modifiedRepMaxValue": modified_rep_max_value,
                "modifiedRepMinValue": modified_rep_min_value,
                "exception": exception,
            }
        )

    return (
        modified_center_value,
        modified_rep_value,
        modified_rep_max_value,
        modified_rep_min_value,
    )


def __get_modified_center_and_rep(
    column: str,
    center: np.ndarray,
    rep: np.ndarray,
    rep_max: np.ndarray,
    rep_min: np.ndarray,
    rule_dict: Dict[str, Dict[str, Any]],
    exception: LogicException,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = get_path_bw_ts(center=center, target=rep, exception=exception)

    modified_center = deepcopy(center)
    modified_rep = deepcopy(rep)
    modified_rep_max = deepcopy(rep_max)
    modified_rep_min = deepcopy(rep_min)

    for idx_in_center, idx_in_rep in path:
        (
            modified_center_value,
            modified_rep_value,
            modified_rep_max_value,
            modified_rep_min_value,
        ) = __get_modified_value_for_center_and_rep(
            column=column,
            center_value=center[idx_in_center],
            rep_value=rep[idx_in_rep],
            rep_max_value=rep_max[idx_in_rep],
            rep_min_value=rep_min[idx_in_rep],
            rule_dict=rule_dict,
            exception=exception,
        )

        modified_center[idx_in_center] = modified_center_value
        modified_rep[idx_in_rep] = modified_rep_value
        modified_rep_max[idx_in_rep] = modified_rep_max_value
        modified_rep_min[idx_in_rep] = modified_rep_min_value

    return modified_center, modified_rep, modified_rep_max, modified_rep_min


def apply_rule_for_model_dicts(
    model_dict: Dict[str, Any],
    mode: str,
    exception: LogicException,
) -> Dict[str, Any]:
    modified_model_dict = {key: None for key in model_dict.keys()}

    if mode == "d_max":
        center_dict = model_dict.get("centerDict")
        rep_dict = model_dict.get("repDict")
        rep_max_dict = model_dict.get("repMaxDict")
        rep_min_dict = model_dict.get("repMinDict")

        modified_center_dict = {column: None for column in center_dict.keys()}
        modified_rep_dict = {column: None for column in rep_dict.keys()}
        modified_rep_max_dict = {column: None for column in rep_max_dict.keys()}
        modified_rep_min_dict = {column: None for column in rep_min_dict.keys()}

        for column in modified_center_dict.keys():
            (
                modified_center,
                modified_rep,
                modified_rep_max,
                modified_rep_min,
            ) = __get_modified_center_and_rep(
                column=column,
                center=center_dict.get(column),
                rep=rep_dict.get(column),
                rep_max=rep_max_dict.get(column),
                rep_min=rep_min_dict.get(column),
                rule_dict=RULES_DICT.get(column),
                exception=exception,
            )

            modified_center_dict.update({column: np.array(modified_center)})
            modified_rep_dict.update({column: np.array(modified_rep)})
            modified_rep_max_dict.update({column: np.array(modified_rep_max)})
            modified_rep_min_dict.update({column: np.array(modified_rep_min)})

        check_all_values_not_None_in_dict(modified_center_dict, exception)
        check_all_values_not_None_in_dict(modified_rep_dict, exception)
        check_all_values_not_None_in_dict(modified_rep_max_dict, exception)
        check_all_values_not_None_in_dict(modified_rep_min_dict, exception)

        modified_model_dict.update(
            {
                "centerDict": modified_center_dict,
                "repDict": modified_rep_dict,
                "repMaxDict": modified_rep_max_dict,
                "repMinDict": modified_rep_min_dict,
            }
        )

    check_all_values_not_None_in_dict(modified_model_dict, exception)
    return modified_model_dict


def predict_by_rule(
    center: np.ndarray,
    target: np.ndarray,
    column: str,
    exception: LogicException,
) -> Dict[str, bool]:
    rule_dict: Dict[str, Dict[str, Any]] = RULES_DICT.get(column)

    pred_by_rule_dict: Dict[str, bool] = {}
    for rule_name, rule_info in rule_dict.items():
        pred: bool = getattr(sys.modules[__name__], f"predict_by_{rule_name}")(
            {
                "center": center,
                "target": target,
                "column": column,
                "rule_info": rule_info,
                "exception": exception,
            }
        )
        pred_by_rule_dict.update({rule_name, pred})

    return pred_by_rule_dict
