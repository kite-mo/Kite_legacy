from typing import Any, Dict, List, Tuple, Union

import numpy as np

from exceptions.logic import LogicException
from utils.utils import get_path_bw_ts


__all__: List[str] = [
    'apply_if_in_0_for_center_and_rep',
    'apply_if_in_100_for_center_and_rep'
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
        exception: 어떤 값으로 할 지에 대한 정보가 없음
        exception: if_in_0인데 대표값 기준으로 하라고 함
        exception: 'by'가 'avg'일 때, 상한 관련 값이 없음
        exception: 'by'가 'avg'일 때, 하한 관련 값이 없음
        exception: 'by'가 'given'일 때, 상한 관련 값이 없음
        exception: 'by'가 'given'일 때, 하한 관련 값이 없음
        exception: 'by'는 'avg' 혹은 'given'이어야 함
        exception: 포함 관계에 대한 이야기가 없음
        exception: 상한 관련 포함 관계에 대한 이야기가 없음
        exception: 하한 관련 포함 관계에 대한 이야기가 없음
    """
    if 'by' not in rule_info.keys():
        message: str = f'Got \'{rule_name}\' rule for {column},'
        message += ' but there is no \'by\' in rule.'
        raise exception(message)

    if rule_info.get('by') == 'avg':
        if rule_name == 'if_in_0':
            message: str = f'Got \'{rule_name}\' rule for {column} by average,'
            message += 'which is contradicted to EHM.'
            raise exception(message)
        if 'plus_value' not in rule_info.keys() and 'plus_percent_value' not in rule_info.keys():
            message: str = f'Got \'{rule_name}\' rule for {column} by average,'
            message += ' but neither \'plus_value\' nor \'plus_percent_value\' exists in rule.'
            raise exception(message)
        if 'minus_value' not in rule_info.keys() and 'minus_percent_value' not in rule_info.keys():
            message: str = f'Got \'{rule_name}\' rule for {column} by average,'
            message += ' but neither \'minus_value\' nor \'minus_percent_value\' exists in rule.'
            raise exception(message)
    elif rule_info.get('by') == 'given':
        if (
            'upper_value' not in rule_info.keys() and
            'plus_value' not in rule_info.keys() and
            'plus_percent_value' not in rule_info.keys()
        ):
            message: str = f'Got \'{rule_name}\' rule for {column},'
            message += ' but there is no neither \'upper_value\''
            message += ' , \'plus_value\' nor \'plus_percent_value\''
            raise exception(message)
        if (
            'lower_value' not in rule_info.keys() and
            'minus_value' not in rule_info.keys() and
            'minus_percent_value' not in rule_info.keys()
        ):
            message: str = f'Got \'if_in_100\' rule for {column},'
            message += ' but there is no neither \'lower_value\''
            message += ' , \'minus_value\' nor \'minus_percent_value\''
            raise exception(message)
    else:
        message: str = f'Got \'{rule_name}\' rule for {column},'
        message += ' \'by\' should be either \'given\' or \'avg\'.'
        message += f'(Got {rule_info.get("by")})'
        raise exception(message)

    if 'equal' not in rule_info.keys():
        message: str = f'Got \'if_in_0\' rule for {column},'
        message += ' but there is no \'equal\' constraint'
        raise exception(message)

    if 'upper' not in rule_info.get('equal'):
        message: str = f'Got \'if_in_0\' rule for {column},'
        message += ' but there is no \'upper\' in \'equal\''
        raise exception(message)

    if 'lower' not in rule_info.get('equal'):
        message: str = f'Got \'if_in_0\' rule for {column},'
        message += ' but there is no \'lower\' in \'equal\''
        raise exception(message)


def __impossible_case(
    lower_limit: float,
    upper_limit: float,
    min_value: float,
    rep_value: float,
    max_value: float,
    exception: LogicException
):
    """
        불가능한 상황이라 raise

    Args:
        lower_limit (float): 하한값
        upper_limit (float): 상한값
        min_value (float): 최소값
        rep_value (float): 대표값
        max_value (float): 최대값
        exception (LogicException): raise할 exception

    Raises:
        exception: 불가능한 상황으로 파악됨
    """
    message: str = f'Is it possible? (Lower limit: {lower_limit};'
    message += f' Upper limit: {upper_limit};'
    message += f' Minimum value: {min_value};'
    message += f' Rep value: {rep_value};'
    message += f' Maximum value: {max_value})'
    raise exception(message)


### Rules for 'if_in_0'
def apply_if_in_0_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    """
    'if_in_0' rule이 적용을 위해 center, rep 관련 value 수정

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Raises:
        exception: 모든 값들이 하한과 상한 사이임 (하한 포함, 상한 포함)
        exception: 모든 값들이 하한과 상한 사이임 (하한 미포함, 상한 포함)
        exception: 모든 값들이 하한과 상한 사이임 (하한 포함, 상한 미포함)
        exception: 모든 값들이 하한과 상한 사이임 (하한 미포함, 상한 미포함)

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
        rule_name='if_in_0',
        column=column,
        exception=exception
    )

    if 'upper_value' in rule_info.keys():
        upper_limit: float = rule_info.get('upper_value')
    elif 'plus_value' in rule_info.keys():
        upper_limit: float = rule_info.get('value') + rule_info.get('plus_value')
    else:
        upper_limit: float = rule_info.get('value')
        upper_limit *= 1 + rule_info.get('plus_value') / 100

    if 'lower_value' in rule_info.keys():
        lower_limit: float = rule_info.get('lower_value')
    elif 'minus_value' in rule_info.keys():
        lower_limit: float = rule_info.get('value') + rule_info.get('minus_value')
    else:
        lower_limit: float = rule_info.get('value')
        lower_limit *= 1 - rule_info.get('minus_percent_value') / 100

    upper_equal: float = rule_info.get('equal').get('upper')
    lower_equal: float = rule_info.get('equal').get('lower')

    # 총 10가지 경우의 수
    # 1. min, rep, max value 모두 하한보다 작은 경우 (문제 없음)
    if (
        (lower_equal and (modified_rep_max_value < lower_limit)) or
        (not lower_equal and (modified_rep_max_value <= lower_limit))
    ):
        pass
    # 2, 3. min, rep value는 하한보다 작음
    elif (
        (lower_equal and (modified_rep_value < lower_limit)) or
        (not lower_equal and (modified_rep_value <= lower_limit))
    ):
        # 2. min, rep value는 하한보다 작으나 max value는 하한과 상한 사이인 경우
        if (
            (upper_equal and (modified_rep_max_value <= upper_limit))
            or (not upper_equal and (modified_rep_max_value < upper_limit))
        ):
            modified_rep_max_value = lower_limit
        # 3. min, rep value는 하한보다 작으나 max value는 상한보다 큰 경우 (front에게 말해야 함)
        elif (
            (upper_equal and (modified_rep_max_value > upper_limit))
            or (not upper_equal and (modified_rep_max_value >= upper_limit))
        ):
            pass
        else:
            __impossible_case(
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                min_value=modified_rep_min_value,
                rep_value=modified_rep_value,
                max_value=modified_rep_max_value,
                exception=exception
            )
    # 4, 5, 6. min value는 하한보다 작음
    elif (
        (lower_equal and (modified_rep_min_value < lower_limit))
        or (not lower_equal and (modified_rep_min_value <= lower_limit))
    ):
        # 4. min value는 하한보다 작으나 rep, max value는 하한과 상한 사이인 경우
        if (
            (lower_equal and upper_equal and
             (lower_limit <= modified_rep_value) and
             (modified_rep_max_value <= upper_limit)) or
            (not lower_equal and upper_equal and
             (lower_limit < modified_rep_value) and
             (modified_rep_max_value <= upper_limit)) or
            (lower_equal and not upper_equal and
             (lower_limit <= modified_rep_value) and
             (modified_rep_max_value < upper_limit)) or
            (not lower_equal and not upper_equal and
             (lower_limit < modified_rep_value) and
             (modified_rep_max_value <= upper_limit))
        ):
            modified_rep_value = lower_limit
            modified_rep_max_value = lower_limit
        # 5. min value는 하한보다 작으나 rep value는 하한과 상한 사이이며 max value는 상한보다 큰 경우 (front에게 말해야 함)
        elif (
            (lower_equal and upper_equal and
             (lower_limit <= modified_rep_value <= upper_limit)
             and (modified_rep_max_value > upper_limit)) or
            (not lower_equal and upper_equal and
             (lower_limit < modified_rep_value <= upper_limit)
             and (modified_rep_max_value > upper_limit)) or
            (lower_equal and not upper_equal and
             (lower_limit <= modified_rep_value < upper_limit)
             and (modified_rep_max_value >= upper_limit)) or
            (not lower_equal and not upper_equal and
             (lower_limit < modified_rep_value <= upper_limit)
             and (modified_rep_max_value >= upper_limit))
        ):
            if modified_rep_value - lower_limit < upper_limit - modified_rep_value:
                modified_rep_value = lower_limit
            else:
                modified_rep_value = upper_limit
        # 6. min value는 하한보다 작으나 rep, max value는 상한보다 큰 경우 (front에게 말해야 함)
        elif (
            (upper_equal and (modified_rep_value > upper_limit)) or
            (not upper_equal and (modified_rep_value >= upper_limit))
        ):
            pass
        else:
            __impossible_case(
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                min_value=modified_rep_min_value,
                rep_value=modified_rep_value,
                max_value=modified_rep_max_value,
                exception=exception
            )
    # 7, 8, 9. min value는 하한과 상한 사이임.
    elif (
        (lower_equal and upper_equal and
         (lower_limit <= modified_rep_min_value <= upper_limit)) or
        (not lower_equal and upper_equal and
         (lower_limit < modified_rep_min_value <= upper_limit)) or
        (lower_equal and not upper_equal and
         (lower_limit <= modified_rep_min_value < upper_limit)) or
        (not lower_equal and not upper_equal and
         (lower_limit < modified_rep_min_value < upper_limit))
    ):
        # 7. min, rep, max value가 모두 하한과 상한 사이인 경우
        if (
            upper_equal and lower_equal
            and (lower_limit <= modified_rep_max_value <= upper_limit)
        ):
            message: str = f'Got \'if_in_0\' rule for {column} by given range,'
            message += ' but both minimum value and maximum value is in the given range.'
            message += f' (Given: [{lower_limit}, {upper_limit}];'
            message += f' Minimum value: {modified_rep_min_value};'
            message += f' Maximum value: {modified_rep_max_value})'
            raise exception(message)
        elif (
            upper_equal and not lower_equal and
            (lower_limit < modified_rep_max_value <= upper_limit)
        ):
            message: str = f'Got \'if_in_0\' rule for {column} by given range,'
            message += ' but both minimum value and maximum value is in the given range.'
            message += f' (Given: ({lower_limit}, {upper_limit}];'
            message += f' Minimum value: {modified_rep_min_value};'
            message += f' Maximum value: {modified_rep_max_value})'
            raise exception(message)
        elif (
            not upper_equal and lower_equal and
            (lower_limit <= modified_rep_max_value < upper_limit)
        ):
            message: str = f'Got \'if_in_0\' rule for {column} by given range,'
            message += ' but both minimum value and maximum value is in the given range.'
            message += f' (Given: [{lower_limit}, {upper_limit});'
            message += f' Minimum value: {modified_rep_min_value};'
            message += f' Maximum value: {modified_rep_max_value})'
            raise exception(message)
        elif (
            not upper_equal and not lower_equal and
            (lower_limit < modified_rep_max_value < upper_limit)
        ):
            message: str = f'Got \'if_in_0\' rule for {column} by given range,'
            message += ' but both minimum value and maximum value is in the given range.'
            message += f' (Given: ({lower_limit}, {upper_limit});'
            message += f' Minimum value: {modified_rep_min_value};'
            message += f' Maximum value: {modified_rep_max_value})'
            raise exception(message)
        # 8. min, rep value는 하한과 상한 사이이나 max value는 상한보다 큰 경우
        elif (
            (upper_equal and (modified_rep_value <= upper_limit) and
             (modified_rep_max_value > upper_limit)) or
            (not upper_equal and (modified_rep_value < upper_limit) and
             (modified_rep_max_value >= upper_limit))
        ):
            modified_rep_value = upper_limit
            modified_rep_min_value = upper_limit
        # 9. min value는 하한과 상한 사이이나 rep, max value는 상한보다 큰 경우
        elif (
            (upper_equal and modified_rep_value > upper_limit) or
            (not upper_equal and modified_rep_value >= upper_limit)
        ):
            modified_rep_min_value = upper_limit
        else:
            __impossible_case(
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                min_value=modified_rep_min_value,
                rep_value=modified_rep_value,
                max_value=modified_rep_max_value,
                exception=exception
            )
    # 10. min, rep, max value 모두 상한보다 큰 경우 (문제 없음)
    elif (
        (upper_equal and (modified_rep_min_value > upper_limit)) or
        (not upper_equal and (modified_rep_max_value >= upper_limit))
    ):
        pass
    else:
        __impossible_case(
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            min_value=modified_rep_min_value,
            rep_value=modified_rep_value,
            max_value=modified_rep_max_value,
            exception=exception
        )

    return modified_center_value, modified_rep_value, modified_rep_max_value, modified_rep_min_value


def predict_by_if_in_0(
    kwargs: Dict[Any, Any]
) -> bool:
    """
        'if_in_0' rule이 적용하여 predict

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Returns:
        bool: 'if_in_0' rule에 만족하는지 여부
    """
    __check_kwargs_contains_all_for_prediction(kwargs=kwargs)
    target: np.ndarray = kwargs.get('target')
    column: str = kwargs.get('column')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='if_in_0',
        column=column,
        exception=exception
    )

    if 'upper_value' in rule_info.keys():
        upper_limit: float = rule_info.get('upper_value')
    elif 'plus_value' in rule_info.keys():
        upper_limit: float = rule_info.get('value') + rule_info.get('plus_value')
    else:
        upper_limit: float = rule_info.get('value')
        upper_limit *= 1 + rule_info.get('plus_value') / 100

    if 'lower_value' in rule_info.keys():
        lower_limit: float = rule_info.get('lower_value')
    elif 'minus_value' in rule_info.keys():
        lower_limit: float = rule_info.get('value') + rule_info.get('minus_value')
    else:
        lower_limit: float = rule_info.get('value')
        lower_limit *= 1 - rule_info.get('minus_percent_value') / 100

    upper_equal: float = rule_info.get('equal').get('upper')
    lower_equal: float = rule_info.get('equal').get('lower')

    pred: Union[None, np.ndarray] = None
    if lower_equal and upper_equal:
        pred = np.array([lower_limit <= x <= upper_limit for x in target])
    elif not lower_equal and upper_equal:
        pred = np.array([lower_limit < x <= upper_limit for x in target])
    elif lower_equal and not upper_equal:
        pred = np.array([lower_limit <= x < upper_limit for x in target])
    else:
        pred = np.array([lower_limit < x < upper_limit for x in target])

    return np.sum(pred) > 0


def get_score_for_if_in_0(
    pred: bool
) -> float:
    """
        'constant_list' rule에 맞게 scoring

    Args:
        pred (bool): 'if_in_0' rule에 만족하는지 여부

    Returns:
        float: 'if_in_0' rule에 만족하면 0점, 아니면 100점
    """
    if pred:
        score: float = 0
    else:
        score: float = 1
    return score
    


### Rules for 'if_in_100'
def apply_if_in_100_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    """
        'if_in_100' rule이 적용을 위해 center, rep 관련 value 수정

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
        rule_name='if_in_100',
        column=column,
        exception=exception
    )

    if 'upper_value' in rule_info.keys():
        upper_limit: float = rule_info.get('upper_value')
    else:
        if rule_info.get('by') == 'given':
            avg: float = rule_info.get('value')
        else:
            avg: float = modified_center_value

        if 'plus_value' in rule_info.keys():
            upper_limit: float = avg + rule_info.get('plus_value')
        else:
            upper_limit: float = avg
            upper_limit *= 1 + rule_info.get('plus_value') / 100

    if 'lower_value' in rule_info.keys():
        lower_limit: float = rule_info.get('lower_value')
    else:
        if rule_info.get('by') == 'given':
            avg: float = rule_info.get('value')
        else:
            avg: float = modified_center_value
        if 'minus_value' in rule_info.keys():
            lower_limit: float = avg - rule_info.get('minus_value')
        else:
            lower_limit: float = avg
            lower_limit *= 1 - rule_info.get('minus_percent_value') / 100

    # 1. rep value가 하한과 상한 사이에 없는 경우
    if (
        (modified_rep_value < lower_limit) or
        (modified_rep_value > upper_limit)
    ):
        modified_rep_value = (lower_limit + upper_limit) / 2

    # 2. min value가 하한보다 큰 경우
    modified_rep_min_value = min(modified_rep_min_value, lower_limit)

    # 3. max value가 상한보다 작은 경우
    modified_rep_max_value = max(modified_rep_max_value, upper_limit)

    return modified_center_value, modified_rep_value, modified_rep_max_value, modified_rep_min_value


def predict_by_if_in_100(
    kwargs: Dict[Any, Any]
) -> bool:
    """
        'if_in_0' rule이 적용하여 predict

    Args:
        kwargs (Dict[Any, Any]): rule이 받은 parameter들

    Returns:
        bool: 'if_in_0' rule에 만족하는지 여부
    """
    __check_kwargs_contains_all_for_prediction(kwargs=kwargs)
    center: np.ndarray = kwargs.get('center')
    target: np.ndarray = kwargs.get('target')
    column: str = kwargs.get('column')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='if_in_0',
        column=column,
        exception=exception
    )

    if 'upper_value' in rule_info.keys():
        upper_limit: np.ndarray = np.array(
            [rule_info.get('upper_value') for _ in range(target.shape[0])]
        )
    else:
        if rule_info.get('by') == 'given':
            avg: np.ndarray = np.array(
                [rule_info.get('value') for _ in range(target.shape[0])]
            )
        else:
            path: List[Tuple[int, int]] = get_path_bw_ts(
                center=center,
                target=target,
                exception=exception,
            )
            avg_list: List[List[float]] = [[] for _ in range(target.shape[0])]
            for center_idx, target_idx in path:
                avg_list[target_idx].append(center[center_idx])
            avg: np.array = np.array([np.mean(avg[i]) for i in range(target.shape[0])])

        if 'plus_value' in rule_info.keys():
            upper_limit: float = avg + rule_info.get('plus_value')
        else:
            upper_limit: float = avg
            upper_limit *= 1 + rule_info.get('plus_value') / 100

    if 'lower_value' in rule_info.keys():
        lower_limit: float = rule_info.get('lower_value')
    else:
        if rule_info.get('by') == 'given':
            avg: float = rule_info.get('value')
        else:
            avg: float = modified_center_value
        if 'minus_value' in rule_info.keys():
            lower_limit: float = avg - rule_info.get('minus_value')
        else:
            lower_limit: float = avg
            lower_limit *= 1 - rule_info.get('minus_percent_value') / 100
