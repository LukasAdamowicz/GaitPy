import sys
import pytest

if sys.version_info > (3, 7):
    from importlib import resources
else:
    import importlib_resources


@pytest.fixture(scope='module')
def demo_data():
    if sys.version_info > (3, 7):
        with resources.path('gaitpy.demo', 'demo_data.csv') as path_:
            path = path_
    else:
        path = importlib_resources.files('gaitpy.demo') / 'demo_data.csv'
    
    return path
