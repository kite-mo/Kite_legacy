import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')


from scoring.scoring import scoring

__all__ = ['scoring']