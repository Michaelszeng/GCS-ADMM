from pydrake.all import (
    MathematicalProgram, 
    Solve, 
    Constraint,
)

import numpy as np
import pandas as pd
import sys
import os
import time

from utils import *

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test1 import As, bs


V, E = build_graph(As, bs)
print(V)
print(E)