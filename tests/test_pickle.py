import pickle
import pysages
import pysages.methods
import pysages.colvars
import inspect
import numpy as np

pi = np.pi


METHODS_ARGS = {
    "HarmonicBias": {"cvs": [pysages.colvars.Component([0], 0)], "kspring": 15.0, "center": 0.7},
    "SamplingMethod": {
        "cvs": [pysages.colvars.Component([0], 0)],
    },
    "ABF": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
    },
    "ANN": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "topology": (15, 23),
        "kT": 1.0,
    },
    "FFS": {
        "cvs": [pysages.colvars.Component([0], 0)],
    },
    "FUNN": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "topology": (15, 23),
    },
    "UmbrellaIntegration": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "kspring": 15.0,
        "center": [0.0, 0.7],
        "hist_periods": 10,
    },
    "Metadynamics": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "height": 1.0,
        "sigma": 5.0,
        "stride": 7.0,
        "ngaussians": 5,
        "deltaT": 0.1,
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "kB": 614.0,
    },
    "HistogramLogger": {
        "period": 1,
        "offset": 1,
    },
    "MetaDLogger": {
        "hills_file": "tmp.txt",
        "log_period": 158,
    },
}


def test_pickle_methods():
    # Iterate all methods of pysages
    for key, pyclass in inspect.getmembers(pysages.methods, inspect.isclass):
        pickle.dumps(pyclass)
        try:
            obj = pyclass(**METHODS_ARGS[key])
        # Filter abstract classes
        except TypeError:
            pass
        else:
            try:
                pickle.dumps(obj)
            except Exception as error:
                print(key)
                raise error


COLVAR_ARGS = {
    "Angle": {"indices": [0, 1, 2]},
    "DihedralAngle": {"indices": [0, 1, 2, 3]},
    "RadiusOfGyration": {"indices": [0, 1, 2, 3]},
    "PrincipalMoment": {"indices": [0, 1, 2, 3], "axis": 0},
    "Asphericity": {"indices": [0, 1, 2, 3]},
    "Acylindricity": {"indices": [0, 1, 2, 3]},
    "ShapeAnisotropy": {"indices": [0, 1, 2, 3]},
    "Component": {"indices": [0, 1, 2, 3], "axis": 0},
    "Distance": {"indices": [0, 1]},
}


def test_pickle_colvars():
    # Iterate all collective variables of PySAGES
    for key, pyclass in inspect.getmembers(pysages.colvars, inspect.isclass):
        pickle.dumps(pyclass)
        try:
            obj = pyclass(**COLVAR_ARGS[key])
        # Filter abstract classes
        except TypeError:
            pass
        else:
            try:
                pickle.dumps(obj)
            except Exception as error:
                print(key)
                raise error
