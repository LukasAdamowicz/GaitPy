import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from gaitpy.signal_features import *

def test__signal_features():
    # Set source and destination directories
    src = os.path.abspath('../gaitpy/demo') + "/demo_data.csv"

    # Run gaitpy
    sample_rate = 50  # hertz
    subject_height = 177  # centimeters
    obtained_classify_bouts, obtained_gait_features = run_gaitpy(src, sample_rate, subject_height)

    # Confirm expected results
    expected_classify_bouts = pd.read_hdf(os.path.abspath('../gaitpy/demo') + '/demo_classify_bouts.h5')
    assert_frame_equal(expected_classify_bouts, obtained_classify_bouts)

    expected_gait_features = pd.read_csv(os.path.abspath('../gaitpy/demo') + '/demo_gait_features.csv')
    expected_gait_features['bout_start_time'] = pd.to_datetime(expected_gait_features['bout_start_time'],
                                                               format='%Y-%m-%d %H:%M:%S.%f')
    assert_frame_equal(expected_gait_features, obtained_gait_features)