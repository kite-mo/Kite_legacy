from typing import List

import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

from prediction.prediction import prediction


__all__: List[str] = ['prediction']