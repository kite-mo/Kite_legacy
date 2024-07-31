from typing import Dict, List, Tuple, Union, Any
import numpy as np

from tslearn.metrics import dtw, dtw_path
from tslearn.barycenters import dtw_barycenter_averaging, euclidean_barycenter

from exceptions.logic import LogicException
from utils.variable import dist_metric, dtw_metric_params, valid_dist_method_list, representative_mode_list


def get_distance_bw_ts(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    exception: LogicException,
    dist_method: str = dist_metric,
    dist_constraint: Union[None, Dict[str, Any]] = dtw_metric_params,
) -> float:
    """
        두 시계열 데이터 사이의 거리 구하기

    Args:
        x (Union[List[float], np.ndarray]): 시계열 데이터
        y (Union[List[float], np.ndarray]): 시계열 데이터
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체
        dist_method (str, optional): 거리를 구하는 방식. Defaults to dist_metric.
        dist_constraint (Union[None, Dict[str, Any]], optional): 거리를 구할 때의 제약. Defaults to dtw_metric_params.

    Raises:
        exception: 구현되지 않은 방식으로 거리를 구하려고 함.
        exception: 거리를 구했는데 float 값이 아님

    Returns:
        float: 두 시계열 데이터 사이의 거리
    """
    
    x = np.array(x)
    y = np.array(y)
    
    if dist_method == 'dtw':
        dist = dtw(
            x, y, 
            global_constraint=dist_constraint.get("global_constraint"), 
            sakoe_chiba_radius = dist_constraint.get("sakoe_chiba_radius")
        )
    elif dist_method == 'euclidean':
        dist = np.sum(np.abs(x - y))
    else:
        raise exception(f'\'dist_method\' should be one of {valid_dist_method_list}. (Got {dist_method})')
        
    dist = float(dist)
    if np.isnan(dist):
        raise exception('Distance should be float.')
    
    return dist


def get_path_bw_ts(
    center: Union[List[float], np.ndarray],
    target: Union[List[float], np.ndarray],
    exception: LogicException,
    dist_method: str = dist_metric,
    dist_constraint: Union[None, Dict[str, Any]] = dtw_metric_params,
) -> List[Tuple[int, int]]:
    """
        두 시계열 데이터 사이의 매칭되는 path 구하기
        
    Args:
        center (Union[List[float], np.ndarray]): 시계열 데이터
        target (Union[List[float], np.ndarray]): 시계열 데이터
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체
        dist_method (str, optional): 거리를 구하는 방식. Defaults to dist_metric.
        dist_constraint (Union[None, Dict[str, Any]], optional): 거리를 구할 때의 제약. Defaults to dtw_metric_params.

    Raises:
        exception: 구현되지 않은 방식으로 거리를 구하려고 함.

    Returns:
        List[Tuple[int, int]]: 두 시계열 데이터 사이의 매칭되는 path
    """
    
    center = np.array(center)
    target = np.array(target)
    
    if dist_method == 'dtw':
        path, _ = dtw_path(
            center, target, 
            global_constraint=dist_constraint.get("global_constraint"), 
            sakoe_chiba_radius = dist_constraint.get("sakoe_chiba_radius")
        )
    elif dist_method == 'euclidean':
        path = [(idx, idx) for idx in range(len(target))]
    else:
        raise exception(f'\'dist_method\' should be one of {valid_dist_method_list}. (Got {dist_method})')
        
    return path


def get_ts_barycenter(
    values_list: List[Union[List[float], np.ndarray]],
    exception: LogicException,
    dist_method: str = dist_metric,
    dist_constraint: Union[None, Dict[str, Any]] = dtw_metric_params,
) -> np.ndarray:
    """
        시계열 데이터들의 무게 중심 구하기

    Args:
        values_list (List[Union[List[float], np.ndarray]]): 시계열 데이터의 리스트
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체
        dist_method (str, optional): 거리를 구하는 방식. Defaults to dist_metric.
        dist_constraint (Union[None, Dict[str, Any]], optional): 거리를 구할 때의 제약. Defaults to dtw_metric_params.

    Returns:
        np.ndarray: 시계열 데이터들의 무게 중심
    """
    
    barycenter = None
    
    if dist_method == 'dtw':
        barycenter = dtw_barycenter_averaging(values_list, metric_params=dist_constraint)
    elif dist_method == 'euclidean':
        barycenter = euclidean_barycenter(values_list)
    else:
        exception(f'\'dist_method\' should be one of {valid_dist_method_list}. (Got {dist_method})')
    
    return barycenter


def get_representative_value(
    values: Union[List[float], np.ndarray],
    if_len_zero_value: Union[None, float],
    representative_mode: str,
    exception: LogicException,
) -> float:
    """
        값들 사이의 대표값 구하기

    Args:
        values (Union[List[float], np.ndarray]): 값의 리스트
        if_len_zero_value (Union[None, float]): values의 길이가 0일 때, 대체할 값
        representative_mode (str): 어떤 대표값을 구할 것인지
        exception (LogicException): 어디에서 난 오류인지 구분하기 위한 예외 처리 객체

    Raises:
        exception: 값의 리스트의 길이가 0인데, 대체할 값이 없음
        exception: 구현되지 않은 방식으로 대표값을 구하려고 함

    Returns:
        float: 대표값
    """
    if len(values) == 0:
        if if_len_zero_value is None:
            raise exception(f'Got length 0 \'values\' but \'if_len_zero_value\' is None.')
        representative_value = if_len_zero_value
    elif representative_mode == 'max':
        representative_value = np.max(values)
    elif representative_mode == 'min':
        representative_value = np.min(values)
    elif representative_mode == 'median':
        representative_value = np.median(values)
    elif representative_mode == 'std':
        representative_value = np.std(values)
    else:
        raise exception(f'Only representative value in {representative_mode_list} can be calculated. (Got {representative_mode})')
    
    return representative_value