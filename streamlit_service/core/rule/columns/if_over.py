from typing import Any, Dict, List, Tuple, Union

import numpy as np

from exceptions.logic import LogicException


__all__ = ['apply_if_over_0_for_center_and_rep', 'apply_if_over_100_for_center_and_rep']


def __check_rule_contains_all(
    rule_info: Dict[str, Any],
    rule_name: str,
    column: str,
    exception: LogicException,
):
    """_summary_

    Args:
        rule_info (Dict[str, Any]): _description_
        rule_name (str): _description_
        column (str): _description_
        exception (LogicException): _description_

    Raises:
        exception: _description_
        exception: _description_
        exception: _description_
        exception: _description_
    """
    if 'by' not in rule_info.keys():
        message: str = f'Got \'{rule_name}\' rule for {column},'
        message += ' but there is no \'by\' in rule.'
        raise exception(message)

    if rule_info.get('by') == 'given':
        if 'value' not in rule_info.keys():
            message: str = f'Got \'{rule_name}\' rule for {column} by given value,'
            message += ' but there is no \'value\' in rule.'
            raise exception(message)
    elif rule_info.get('by') == 'avg':
        if 'plus_value' not in rule_info.keys() and 'plus_percent_value' not in rule_info.keys():
            message: str = f'Got \'{rule_name}\' rule for {column} by average,'
            message += ' but neither \'plus_value\' nor \'plus_percent_value\' exists in rule.'
            raise exception(message)
    else:
        message: str = f'Got \'{rule_name}\' rule for {column},'
        message += ' \'by\' should be either \'given\' or \'avg\'.'
        message += f'(Got {rule_info.get("by")})'
        raise exception(message)


### Rules for 'if_over_0'
def apply_if_over_0_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    """_summary_

    Args:
        kwargs (Dict[Any, Any]): _description_

    Returns:
        Tuple[float, float, float, float]: _description_
    """
    column: str = kwargs.get('column')
    modified_center_value: float = kwargs.get('modifiedCenterValue')
    modified_rep_value: float = kwargs.get('modifiedRepValue')
    modified_rep_max_value: float = kwargs.get('modifiedRepMaxValue')
    modified_rep_min_value: float = kwargs.get('modifiedRepMinValue')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='if_over_0',
        column=column,
        exception=exception
    )

    # 'if_over_0'의 경우, rule value는 상한을 의미
    if rule_info.get('by') == 'given':
        upper_limit = rule_info.get('value')
    elif rule_info.get('by') == 'avg':
        if 'plus_value' in rule_info.keys():
            upper_limit = modified_center_value + rule_info.get('plus_value')
        else:
            upper_limit = modified_center_value * (1 + rule_info.get('plus_percent_value') / 100)

    modified_rep_value = min(modified_rep_value, upper_limit)
    modified_rep_min_value = min(modified_rep_min_value, upper_limit)
    modified_rep_max_value = min(modified_rep_max_value, upper_limit)

    return modified_center_value, modified_rep_value, modified_rep_max_value, modified_rep_min_value


### Rules for 'if_over_100'
def apply_if_over_100_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    """_summary_

    Args:
        kwargs (Dict[Any, Any]): _description_

    Returns:
        Tuple[float, float, float, float]: _description_
    """
    column: str = kwargs.get('column')
    modified_center_value: float = kwargs.get('modifiedCenterValue')
    modified_rep_value: float = kwargs.get('modifiedRepValue')
    modified_rep_max_value: float = kwargs.get('modifiedRepMaxValue')
    modified_rep_min_value: float = kwargs.get('modifiedRepMinValue')
    rule_info: Dict[str, Any] = kwargs.get('rule_info')
    exception: LogicException = kwargs.get('exception')

    __check_rule_contains_all(
        rule_info=rule_info,
        rule_name='if_over_100',
        column=column,
        exception=exception
    )

    # 'if_over_100'의 경우, rule value는 하한을 의미
    if rule_info.get('by') == 'given':
        lower_limit: float = rule_info.get('value')
    elif rule_info.get('by') == 'avg':
        if 'plus_value' in rule_info.keys():
            lower_limit: float = modified_center_value + rule_info.get('plus_value')
        else:
            lower_limit: float = modified_center_value * (1 + rule_info.get('plus_percent_value') / 100)

    modified_rep_max_value = np.inf

    modified_rep_min_value = min(modified_rep_min_value, lower_limit)

    return modified_center_value, modified_rep_value, modified_rep_max_value, modified_rep_min_value
