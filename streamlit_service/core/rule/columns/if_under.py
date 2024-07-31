from typing import Any, Dict, List, Tuple, Union

import numpy as np


__all__ = ['apply_if_under_0_for_center_and_rep', 'apply_if_under_100_for_center_and_rep']


### Rules for 'if_under_0'
def apply_if_under_0_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    
    modified_center_value = kwargs.get('modifiedCenterValue')
    modified_rep_value = kwargs.get('modifiedRepValue')
    modified_rep_max_value = kwargs.get('modifiedRepMaxValue')
    modified_rep_min_value = kwargs.get('modifiedRepMinValue')
    rule_info = kwargs.get('rule_info')
    
    # 'if_under_0'의 경우, rule value는 하한을 의미
    if rule_info.get('by') == 'given':
        lower_limit = rule_info.get('value')
    elif rule_info.get('by') == 'avg':
        if 'minus_value' in rule_info.keys():
            lower_limit = modified_center_value - rule_info.get('minus_value')
        else:
            lower_limit = modified_center_value * (1 - rule_info.get('minus_percent_value') / 100)
        
    if modified_center_value < lower_limit:
        modified_center_value = lower_limit
    
    if modified_rep_value < lower_limit:
        modified_rep_value = lower_limit
    
    if modified_rep_max_value < lower_limit:
        modified_rep_max_value = lower_limit
    
    if modified_rep_min_value < lower_limit:
        modified_rep_min_value = lower_limit
        
    
    return modified_center_value, modified_rep_value, modified_rep_max_value, modified_rep_min_value


### Rules for 'if_under_100'
def apply_if_under_100_for_center_and_rep(
    kwargs: Dict[Any, Any]
) -> Tuple[float, float, float, float]:
    
    modified_center_value = kwargs.get('modifiedCenterValue')
    modified_rep_value = kwargs.get('modifiedRepValue')
    modified_rep_max_value = kwargs.get('modifiedRepMaxValue')
    modified_rep_min_value = kwargs.get('modifiedRepMinValue')
    rule_info = kwargs.get('rule_info')
    
    # 'if_under_100'의 경우, rule value는 상한을 의미
    # ('by' == 'avg')일 수 없음
    upper_limit = rule_info.get('value')
        
    modified_rep_min_value = - np.inf

    if modified_rep_max_value < upper_limit:
        modified_rep_max_value = upper_limit    
    
    return modified_center_value, modified_rep_value, modified_rep_max_value, modified_rep_min_value
