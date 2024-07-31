from typing import Dict, Tuple, List, Union, Any

import pandas as pd

from exceptions.logic import LogicException


__all__ = [
    'check_all_columns_in_df_list',
    'check_all_values_not_None_in_dict',
    'check_one_rule_for_one_column',
    'check_rule_requirements'
]


def check_column_type(
    df: pd.DataFrame,
    time_column: str,
    numeric_columns: List[str],
    exception: LogicException
):
    try:
        df[time_column] = pd.to_datetime(df[time_column])
    except:
        raise exception(f'{time_column} cannot be converted to datetime type.\nYou might write wrong name for time related column.')

    not_numeric_columns = []
    for column in numeric_columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except:
            not_numeric_columns.append(column)
    
    if len(not_numeric_columns) > 0:
        raise exception(f'{not_numeric_columns} are in \'numeric_columns\', but these columns cannot be converted to a numeric value.')


def check_all_columns_in_df_list(
    df_list: List[pd.DataFrame],
    columns_list: List[str],
    exception: LogicException
):
    """
        모든 DataFrame에 모든 column이 있는지 확인

    Args:
        df_list (List[pd.DataFrame]): DataFrame 리스트
        columns_list (List[str]): column 리스트
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: column이 DataFrame에 존재하지 않음
    """
    
    check_dict = {column: [] for column in columns_list}
    
    column_not_exist_flag = False
    for df_idx, df in enumerate(df_list):
        for column in columns_list:
            if column not in df.columns:
                check_dict[column].append(df_idx)
                column_not_exist_flag = True
    
    if column_not_exist_flag:
        if len(df_list) > 1:
            message = "There are some columns which are in \'hm_column_list\' but not in some of DataFrames.\n"
            message += "Column Name: Index for DataFrames in \'df_list\'.\n"
            for column, check_list in check_dict.items():
                if len(check_list) > 0:
                    message += f"\t{column}: {check_list}\n"
        else:
            message = "There are some columns which are in \'hm_column_list\' but not in the DataFrame.\n"
            for column, check_list in check_dict.items():
                if len(check_list) > 0:
                    message += f"\t{column}\n"
            
        raise exception(message)


def check_all_values_not_None_in_dict(
    dict: Dict[Any, Any],
    exception: LogicException
):
    """
        dictionary에 있는 value 중에 None이 있는지 확인

    Args:
        dict (Dict[Any, Any]): 확인할 dictionary
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: dictionary의 어떤 key가 None인 value를 가짐
    """
    none_keys = []
    for key, value in dict.items():
        if value is None:
            none_keys.append(key)
    
    if len(none_keys) > 0:
        raise exception(f'Key(s) in {none_keys} has(have) None value.')
    

def check_one_rule_for_one_column(
    applied_rule: List[Dict[str, Dict[str, Any]]],
    column: str,
    exception: LogicException
):
    """
        column이 rule 내에서 하나의 sensor_list에만 포함되는지 확인

    Args:
        applied_rule (List[Dict[str, Dict[str, Any]]]): column이 sensor_list에 포함된 rule
        column (str): column 명
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: column이 rule 내에서 여러 sensor_list에 포함됨
    """
    if len(applied_rule) > 1:
        raise exception(f'Rule Dict for {column} is more than one. Cannot apply rule for {column}.')


def __check_constant_requirements(
    rule_dict: Dict[str, Dict[str, Any]],
    column: str,
    exception: LogicException
):
    """
        'constant'와 관련된 rule이 EHM에 맞게 설정되어있는지 확인

    Args:
        rule_dict (Dict[str, Dict[str, Any]]): key: rule_name, value: rule_info인 dictionary
        column (str): column 명
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: 'constant' rule을 가지려면 rule_dict에 하나만 있어야 함
        exception: 'value'가 rule_info에 없는 경우, 어떤 값을 기준으로 할 지 알 수가 없음
    """
    if len(rule_dict.keys()) > 1:
        raise exception(f'Got \'constant\' rule for {column}, but got more than one rule. ({list(rule_dict.keys())})')
    if 'value' not in rule_dict.get('constant').keys():
        raise exception(f'Got \'constant\' rule for {column}, but \'value\' is not in rule.')
    
    
def __check_if_over_requirements(
    rule_info:Dict[str, Any],
    rule_name: str,
    column: str,
    exception: LogicException
):
    """
        'if_over'와 관련된 rule이 EHM에 맞게 설정되어있는지 확인

    Args:
        rule_info (Dict[str, Any]): 'if_over'와 관련된 정보
        rule_name (str): 'if_over_0' or 'if_over_100'
        column (str): column 명
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: 'by'가 rule_info에 없는 경우, 어떤 규칙인지 알 수가 없음
        exception: 'by'가 'given'인데 'value'가 rule_info에 없는 경우, 어떤 값을 기준으로 할 지 알 수가 없음
        exception: 'by'가 'avg'인데 'if_over_100'인 경우, EHM과 맞지 않음
        exception: 'by'가 'avg'인데 'plus_value' 혹은 'plus_percent_value'가 rule_info에 없는 경우, 어떤 값을 기준으로 할 지 알 수가 없음
    """
    # 'by'가 없으면 어떤 규칙인지 알 수 없음
    if 'by' not in rule_info.keys():
        raise exception(f'Got \'{rule_name}\' rule for {column}, but \'by\' is not in rule.')
    
    
    # 'by'가 'given'인데 'value'가 없으면 어떤 값을 기준으로 할 지 알 수가 없음
    if rule_info.get('by') == 'given' and 'value' not in rule_info.keys():
        raise exception(f'Got \'{rule_name}\' rule for {column} by given value, but \'value\' is not in rule.')
    
    elif rule_info.get('by') == 'avg':
    # 평균보다 커야 100점이라는 것은 EHM과 맞지 않음
        if rule_name == 'if_over_100':
            raise exception(f'Got \'{rule_name}\' rule for {column} by avg, which is contradicted to EHM.')
    # 'by'가 'avg'인데 'plus_value' 혹은 'plus_percent_value'가 없으면 어떤 값을 기준으로 할 지 알 수가 없음
        if ('plus_value' not in rule_info.keys() and 'plus_percent_value' not in rule_info.keys()):
            raise exception(f'Got \'{rule_name}\' rule for {column} by avg, but either \'plus_value\' or \'plus_percent_value\' is not in rule. ({list(rule_info.keys())})')


def __check_if_under_requirements(
    rule_info:Dict[str, Any],
    rule_name: str,
    column: str,
    exception: LogicException
):
    """
        'if_over'와 관련된 rule이 EHM에 맞게 설정되어있는지 확인

    Args:
        rule_info (Dict[str, Any]): 'if_under'와 관련된 정보
        rule_name (str): 'if_under_0' or 'if_under_100'
        column (str): column 명
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: 'by'가 rule_info에 없는 경우, 어떤 규칙인지 알 수가 없음
        exception: 'by'가 'given'인데 'value'가 rule_info에 없는 경우, 어떤 값을 기준으로 할 지 알 수가 없음
        exception: 'by'가 'avg'인데 'if_under_100'인 경우, EHM과 맞지 않음
        exception: 'by'가 'avg'인데 'minus_value' 혹은 'minus_percent_value'가 rule_info에 없는 경우, 어떤 값을 기준으로 할 지 알 수가 없음
    """
    # 'by'가 없으면 어떤 규칙인지 알 수 없음
    if 'by' not in rule_info.keys():
        raise exception(f'Got \'{rule_name}\' rule for {column}, but \'by\' is not in rule.')
    
    
    # 'by'가 'given'인데 'value'가 없으면 어떤 값을 기준으로 할 지 알 수가 없음
    if rule_info.get('by') == 'given' and 'value' not in rule_info.keys():
        raise exception(f'Got \'{rule_name}\' rule for {column} by given value, but \'value\' is not in rule.')
    elif rule_info.get('by') == 'avg':
        # 평균보다 작아야 100점이라는 것은 EHM과 맞지 않음
        if rule_name == 'if_under_100':
            raise exception(f'Got \'{rule_name}\' rule for {column} by avg, which is contradicted to EHM.')
        # 'by'가 'avg'인데 'minus_value' 혹은 'minus_percent_value'가 없으면 어떤 값을 기준으로 할 지 알 수가 없음
        if ('minus_value' not in rule_info.keys() and 'minus_percent_value' not in rule_info.keys()):
            raise exception(f'Got \'{rule_name}\' rule for {column} by avg, but either \'minus_value\' or \'minus_percent_value\' is not in rule. ({list(rule_info.keys())})')


def __check_if_in_requirements(
    rule_info:Dict[str, Any],
    rule_name: str,
    column: str,
    exception: LogicException
):
    """
        'if_in'과 관련된 rule이 EHM에 맞게 설정되어있는지 확인

    Args:
        rule_info (Dict[str, Any]): 'if_in'과 관련된 정보
        rule_name (str): 'if_in_0' or 'if_in_100'
        column (str): column 명
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: 'by'가 rule_info에 없는 경우, 어떤 규칙인지 알 수가 없음
        exception: 'by'가 'given'인데 'lower_value' 혹은 'upper_value'가 rule_info에 없는 경우, 어떤 값을 기준으로 할 지 알 수가 없음
        exception: 'by'가 'avg'인데 'if_in_0'인 경우, EHM과 맞지 않음
        exception: 'by'가 'avg'인데 상한 혹은 하한에 대한 값이 없음
    """
    # 'by'가 없으면 어떤 규칙인지 알 수 없음
    if 'by' not in rule_info.keys():
        raise exception(f'Got \'{rule_name}\' rule for {column}, but \'by\' is not in rule.')
    
    
    # 'by'가 'given'인데 'lower_value' 혹은 'upper_value'가 없으면 어떤 값을 기준으로 할 지 알 수가 없음
    if rule_info.get('by') == 'given' and ('lower_value' not in rule_info.keys() or 'upper_value' not in rule_info.keys()):
        raise exception(f'Got \'{rule_name}\' rule for {column} by given value, but either \'lower_value\' or \'upper_value\' is not in rule.')
    elif rule_info.get('by') == 'avg':
        # 평균을 기준으로 특정 범위의 값들이 0점이라는 것은 EHM과 맞지 않음
        if rule_name == 'if_in_0':
            raise exception(f'Got \'{rule_name}\' rule for {column} by avg, which is contradicted to EHM.')
        plus_exists = ('plus_value' in rule_info.keys() or 'plus_percent_value' in rule_info.keys())
        minus_exists = ('minus_value' in rule_info.keys() or 'minus_percent_value' in rule_info.keys())
        # 상한 혹은 하한에 대한 값이 없음
        if not plus_exists or not minus_exists:
            raise exception(f'Got \'{rule_name}\' rule for {column} by avg, but both \'plus\' related value and \'minus\' related value are not in rule. ({list(rule_info.keys())})')


def check_rule_requirements(
    rule_dicts: Dict[str, Dict[str, Dict[str, Any]]],
    exception: LogicException
):
    """
        각 rule이 EHM에 맞게 설정되어있는지 확인

    Args:
        rule_dicts (Dict[str, Dict[str, Dict[str, Any]]]): key: rule_name, value: rule_info인 dictionary
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체
    """
    for column, rule_dict in rule_dicts.items():
        for rule_name, rule_info in rule_dict.items():
            if rule_name == 'constant':
                __check_constant_requirements(rule_dict=rule_dict, column=column, exception=exception)
            elif 'if_over' in rule_name:
                __check_if_over_requirements(rule_info=rule_info, rule_name=rule_name, column=column, exception=exception)
            elif 'if_under' in rule_name:
                __check_if_under_requirements(rule_info=rule_info, rule_name=rule_name, column=column, exception=exception)
            elif 'if_in' in rule_name:
                __check_if_in_requirements(rule_info=rule_info, rule_name=rule_name, column=column, exception=exception)
            
