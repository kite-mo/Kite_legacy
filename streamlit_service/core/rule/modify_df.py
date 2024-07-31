from typing import Any, Dict, List, Tuple, Union

import numpy as np


__all__: List[str] = ['apply_exclude_for_values', 'apply_include_for_values']


### Rules for 'exclude'
def apply_exclude_for_values(
    kwargs: Dict[Any, Any]
) -> np.ndarray:
    # TODO: apply 'exclude' rule
    return kwargs['modifiedValues']


### Rules for 'include'
def apply_include_for_values(
    kwargs: Dict[Any, Any]
) -> np.ndarray:
    # TODO: apply 'include' rule
    return kwargs['modifiedValues']