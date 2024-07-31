from typing import Any, Dict, List, Tuple, Union

import numpy as np

from exceptions.logic import LogicException
from utils.variable import precision


__all__: List[str] = [
    'apply_constant_for_center_and_rep',
    'apply_constant_list_for_center_and_rep',
    'predict_by_constant',
    'predict_by_constant_list',
    'get_score_for_constant',
    'get_score_for_constant_list'
]


def __check_kwargs_contains_all_for_center_and_rep(
    kwargs: Dict[Any, Any]
):
    """
        kwargs에 모든 요소가 있는지 확인

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Raises:
        ValueError: exception이 없음
        exception: column이 없음
        exception: rule_info가 없음
        exception: center value가 없음
        exception: rep value가 없음
        exception: max value가 없음
        exception: min value가 없음
    """
    if 'exception' not in kwargs.keys():
        raise ValueError('There is no \'exception\' in kwargs.')

    exception: LogicException = kwargs.get('exception')
    if 'column' not in kwargs.keys():
        raise exception('There is no \'column\' in kwargs.')
    if 'rule_info' not in kwargs.keys():
        raise exception('There is no \'rule_info\' in kwargs.')
    if 'modifiedCenterValue' not in kwargs.keys():
        raise exception('There is no \'modifiedCenterValue\' in kwargs.')
    if 'modifiedRepValue' not in kwargs.keys():
        raise exception('There is no \'modifiedRepValue\' in kwargs.')
    if 'modifiedRepMaxValue' not in kwargs.keys():
        raise exception('There is no \'modifiedRepMaxValue\' in kwargs.')
    if 'modifiedRepMinValue' not in kwargs.keys():
        raise exception('There is no \'modifiedRepMinValue\' in kwargs.')


def __check_kwargs_contains_all_for_prediction(
    kwargs: Dict[Any, Any]
):
    """
        kwargs에 모든 요소가 있는지 확인

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Raises:
        ValueError: exception이 없음
        exception: center가 없음
        exception: target이 없음
        exception: column이 없음
        exception: rule_info가 없음
    """
    if 'exception' not in kwargs.keys():
        raise ValueError('There is no \'exception\' in kwargs.')

    exception: LogicException = kwargs.get('exception')
    if 'center' not in kwargs.keys():
        raise exception('There is no \'center\' in kwargs.')
    if 'target' not in kwargs.keys():
        raise exception('There is no \'target\' in kwargs.')
    if 'column' not in kwargs.keys():
        raise exception('There is no \'column\' in kwargs.')
    if 'rule_info' not in kwargs.keys():
        raise exception('There is no \'rule_info\' in kwargs.')


def __check_rule_contains_all(
    rule_info: Dict[str, Any],
    rule_name: str,
    column: str,
    exception: LogicException,
):
    """
        rule에 모든 요소가 있는지 확인

    Args:
        rule_info (Dict[str, Any]): rule 관련 정보
        rule_name (str): rule 이름
        column (str): 센서 명
        exception (LogicException): raise할 exception

    Raises:
        exception: 'constant'인데 기준값이 없음
        exception: 'constant_list'인데 기준값 리스트가 없음
    """
    if rule_name == 'constant' and 'value' not in rule_info.keys():
        message: str = f'Got \'{rule_name}\' rule for {column},'
        message += ' but there is no \'value\'.'
        raise exception(message)

    if rule_name == 'constant_list' and 'value_list' not in rule_info.keys():
        message: str = f'Got \'{rule_name}\' rule for {column},'
        message += ' but there is no \'value_list\'.'
        raise exception(message)


### Rules for constant
def apply_constant_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    """
        'constant' rule이 적용을 위해 center, rep 관련 value 수정

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Returns:
        Tuple[float, float, float, float]: 변경된 center, rep, min, max value
    """
    __check_kwargs_contains_all_for_center_and_rep(kwargs=kwargs)

    column: str = kwargs.get('column')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='constant',
        column=column,
        exception=exception
    )
    constant_value: float = rule_info.get('value')
    return constant_value, constant_value, constant_value, constant_value


def predict_by_constant(
    kwargs: Dict[Any, Any]
) -> bool:
    """
        'constant' rule이 적용하여 predict

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Returns:
        float: 모든 포인트를 constant와 비교한 거리의 합
    """
    __check_kwargs_contains_all_for_prediction(kwargs=kwargs)
    target: np.ndarray = kwargs.get('target')
    column: str = kwargs.get('column')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='constant',
        column=column,
        exception=exception
    )
    constant_value: float = rule_info.get('value')

    pred: float = np.sum(np.abs(target - constant_value))

    return pred <= precision


def get_score_for_constant(
    pred: bool
) -> float:
    """
        'constant' rule에 맞게 scoring

    Args:
        pred (bool): 'constant' rule에 만족하는지 여부

    Returns:
        float: 'constant' rule에 만족하면 100점, 아니면 0점
    """
    if pred:
        score: float = 1
    else:
        score: float = 0
    return score


### Rules for constant_list
def apply_constant_list_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    """
        'constant_list' rule이 적용을 위해 center, rep 관련 value 수정

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Returns:
        Tuple[float, float, float, float]: 변경된 center, rep, min, max value
    """
    __check_kwargs_contains_all_for_center_and_rep(kwargs=kwargs)

    column: str = kwargs.get('column')
    modified_center_value: float = kwargs.get('modifiedCenterValue')
    modified_rep_value: float = kwargs.get('modifiedRepValue')
    modified_rep_max_value: float = kwargs.get('modifiedRepMaxValue')
    modified_rep_min_value: float = kwargs.get('modifiedRepMinValue')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='constant_list',
        column=column,
        exception=exception
    )

    # constant_value_list: List[float] = rule_info.get('value_list')

    return modified_center_value, modified_rep_value, modified_rep_max_value, modified_rep_min_value


def predict_by_constant_list(
    kwargs: Dict[Any, Any]
) -> np.ndarray:
    """
        'constant_list' rule이 적용하여 predict

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Returns:
        float: 'constant_list' rule에 만족하는지 여부
    """
    __check_kwargs_contains_all_for_prediction(kwargs=kwargs)
    target: np.ndarray = kwargs.get('target')
    column: str = kwargs.get('column')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='constant_list',
        column=column,
        exception=exception
    )
    constant_value_list: float = rule_info.get('value_list')

    pred: np.ndarray = np.array([
        x in constant_value_list
        for x in target
    ])

    return np.sum(~pred) == 0


def get_score_for_constant_list(
    pred: bool
) -> float:
    """
        'constant_list' rule에 맞게 scoring

    Args:
        pred (bool): 'constant_list' rule에 만족하는지 여부

    Returns:
        float: 'constant_list' rule에 만족하면 100점, 아니면 0점
    """
    if pred:
        score: float = 1
    else:
        score: float = 0
    return score
