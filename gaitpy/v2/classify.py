"""
Classes for classifying IMU data as gait or not-gait
"""
import pandas as pd
from numpy import require, zeros, sign, ndarray
from numpy.lib.stride_tricks import as_strided
import h5py

from gaitpy.v2.base import _BaseProcess

import signal_features as SF


__all__ = []


class RFGaitClassifier:
    def __init__(self, sampling_frequency, window_length=3.0, step_size=1.0, correct_alignment=False):
        """
        Classify windows of time-series acceleration data as either gait or non-gait. Optionally correct for
        accelerometer axis alignment with vertical, window, compute signal features, and classify as gait using
        a Random Forest model.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency of the time-series data
        window_length : float, optional
            Window length in seconds. Default is 3.0s
        step_size : float, int, optional
            Window step size. If a float between 0 and 1.0, taken as the percentage of a window to skip
            between window centers. If an integer greater or equal to 1, taken as the number of samples to skip
            between window centers. Default is 1.0 (windows with no overlap).
        correct_alignment : bool, optional
            Correct the alignment of the vertical sensor axis. Default is False. Using this correction requires that the
            data from all three axes be provided.

        Examples
        --------
        Step 1.0s (2.0 * 0.5) between window centers. This results in the first half of a window overlapping with the
        previous window, and the second half overlapping with the following window
        >>> rfcls = RFGaitClassifier(50.0, window_length=2.0, step_size=0.5)

        Step 50 samples between window centers
        >>> rfcls = RFGaitClassifier(50.0, window_length=3.0, step_size=50)
        """
        self.fs = sampling_frequency

        # save for str/repr
        if window_length > 0:
            self.window_length = window_length
        else:
            raise ValueError("'window_length' must be positive")
        self.step_size = step_size

        self.win_l = int(window_length * self.fs)
        if isinstance(step_size, float):
            if step_size == 1.0:
                self.step = self.win_l
            elif 0.0 <= step_size < 1.0:
                tmp_ = int(self.win_l * step_size)
                self.step = tmp_ if tmp_ >= 1 else 1
            else:
                raise ValueError('Float step_size must be between 0.0 and 1.0')
        elif isinstance(step_size, int):
            if step_size >= 1:
                self.step = step_size
            else:
                raise ValueError('Integer step_size must be greater than or equal to 1')
        else:
            raise ValueError('step_size must be either a float or integer')

    def predict(self, data, axis=None, indices=None):
        """
        Classify gait

        Parameters
        ----------
        data : {str, dict, numpy.ndarray, pandas.DataFrame}
            A string to an H5 file, a dictionary, ndarray, or DataFrame containing the raw accelerometer data to
            classify as gait/non-gait. If 1D, it is assumed to be the vertical axis, with . If 3D, the `axis` argument is
            required to specify which axis is aligned with vertical. If `correct_alignment=True`, then the input must
            be 3D. See Notes for where the accelerometer data should be located in H5 files or dictionaries.
        axis : int, optional
            Required if providing 3D data, the axis that is best aligned with vertical. Alignment should be specified as
            either positive (axis pointing "up", measured still acceleration -9.8m/s^2) or negative (axis pointing
            "down", measured still acceleration 9.8m/s^2).
        indices : array_like, optional
            Indices for the start and stop of specific days. Only used if data is a ndarray or DataFrame. If providing
            a H5 file path or dictionary, these indices should be stored in the object itself (see Notes).

        Returns
        -------

        Notes
        -----
        If `data` is a H5 file or dictionary, the accelerometer data (as ndarrays) should be located in
        `data['Sensors']['Lumbar']['Accelerometer']`. Indices should be located in
        `data['Processed']['Gait']['Day #']['Indices']` as a 2 element array [start, stop] for each day.

        Examples
        --------
        Providing a H5 file path:
        >>> path = '/folder1/study1/data/data.h5'
        >>> rfcls.predict(path, axis=1)  # use the y axis as aligned with vertical and pointing up

        Providing a 1D pandas.DataFrame
        >>> data = pd.DataFrame(accel_data)
        >>> rfcls.predict(data)

        Providing day start and stop indices for 2 days
        >>> data = accel_data
        >>> indices = np.array([[0, 324000], [324000, 785700]])
        >>> rfcls.predict(data, indices=indices)
        """
        if isinstance(data, pd.DataFrame):
            tmp = data.values

            if tmp.shape[1] == 1:
                acc = tmp.flatten()
            else:
                if axis is None:
                    raise ValueError("'axis' must be specified for non-1d input data.")
                else:
                    acc = tmp[abs(axis)] * sign(axis)

            if indices is not None:
                res = {}
                for i, idx in enumerate(indices):
                    res[f'Day {i+1}'] = self._call(acc[idx[0]:idx[1]])
            else:
                res = self._call(acc)

            return res
        elif isinstance(data, ndarray):
            if data.ndim == 1:
                acc = data
            else:
                if axis is None:
                    raise ValueError("'axis' must be specified for non-1d input data.")
                else:
                    data[abs(axis)] * sign(axis)

            if indices is not None:
                res = {}
                for i, idx in enumerate(indices):
                    res[f'Day {i + 1}'] = self._call(acc[idx[0]:idx[1]])
            else:
                res = self._call(acc)

            return res
        elif isinstance(data, dict):
            if data['Sensors']['Lumbar']['Accelerometer'].ndim == 1:
                acc = data['Sensors']['Lumbar']['Accelerometer']
            else:
                if axis is None:
                    raise ValueError("'axis' must be specified for non-1d input data.")
                else:
                    acc = data['Sensors']['Lumbar']['Accelerometer'][abs(axis)] * sign(axis)

            if 'Processed' in data:
                days = [i for i in data['Processed']['Gait'] if 'Day' in i]
            else:
                days = ['Day 1']

            for day in days:
                try:
                    start, stop = data['Processed']['Gait'][day]['Indices']
                except KeyError:
                    start, stop = 0, acc.size

                # TODO assign to dictionary
                res = self._call(acc[start:stop])

        else:
            with h5py.File(data, 'r') as f:
                if f['Sensors/Lumbar/Accelerometer'].ndim == 1:
                    acc = f['Sensors/Lumbar/Accelerometer'][()]
                else:
                    if axis is None:
                        raise ValueError("'axis' must be specified for non-1d input data.")
                    else:
                        acc = f['Sensors/Lumbar/Accelerometer'][abs(axis)] * sign(axis)

                if 'Processed' in f:
                    days = [i for i in data['Processed/Gait'] if 'Day' in i]
                else:
                    days = ['Day 1']

                for day in days:
                    try:
                        start, stop = data[f'Processed/Gait/{day}/Indices']
                    except KeyError:
                        start, stop = 0, acc.size

                    # TODO assign into h5
                    res = self._call(acc[start:stop])

    def _call(self, vacc):
        # TODO filter

        # window the vertical acceleration
        win_vacc = RFGaitClassifier._get_windowed_view(vacc, self.win_l, self.step, True)

        # set-up the space for the features
        feats = zeros((win_vacc.shape[0], 8))

        # compute the features
        ft_freq = SF.dominant_frequency(win_vacc, self.fs, low_cutoff=0.0, high_cutoff=12.0)

        feats[:, 0] = ft_freq[2]  # dominant frequency ratio
        feats[:, 1] = ft_freq[0]  # dominant frequency value
        feats[:, 2] = SF.mean_cross_rate(win_vacc, axis=1)
        feats[:, 3] = SF.signal_range(win_vacc, axis=1)
        feats[:, 4] = SF.signal_rms(win_vacc, axis=1)
        feats[:, 5] = SF.signal_entropy(win_vacc)
        feats[:, 6] = ft_freq[4]  # spectral entropy
        feats[:, 7] = ft_freq[3]  # spectral flatness


    @staticmethod
    def _get_windowed_view(x, window_length, step_size, ensure_c_contiguity=False):
        """
        Return a moving window view over the data

        Parameters
        ----------
        x : numpy.ndarray
            1- or 2-D array of signals to window. Windows occur along the 0 axis. Must be C-contiguous.
        window_length : int
            Window length/size.
        step_size : int
            Step/stride size for windows - how many samples to step from window center to window center.
        ensure_c_contiguity : bool, optional
            Create a new array with C-contiguity if the passed array is not C-contiguous. This *may* result in the
            memory requirements significantly increasing. Default is False, which will raise a ValueError if `x` is
            not C-contiguous

        Returns
        -------
        x_win : numpy.ndarray
            2- or 3-D array of windows of the original data, of shape (..., L[, ...])
        """
        if not (x.ndim in [1, 2]):
            raise ValueError('Array cannot have more than 2 dimensions.')

        if ensure_c_contiguity:
            x = require(x, requirements=['C'])
        else:
            if not x.flags['C_CONTIGUOUS']:
                raise ValueError("Input array must be C-contiguous.  See numpy.ascontiguousarray")

        if x.ndim == 1:
            nrows = ((x.size - window_length) // step_size) + 1
            n = x.strides[0]
            return as_strided(x, shape=(nrows, window_length), strides=(step_size * n, n), writeable=False)

        else:
            k = x.shape[1]
            nrows = ((x.shape[0] - window_length) // step_size) + 1
            n = x.strides[1]

            new_shape = (nrows, window_length, k)
            new_strides = (step_size * k * n, k * n, n)
            return as_strided(x, shape=new_shape, strides=new_strides, writeable=False)


class SignalFeatureExtractor(_BaseProcess):
    def __init__(self, features='default', window_length=3.0, step_size=1.0, **kwargs):
        """
        Extract features from raw inertial data, given the window length and step size.

        Parameters
        ----------
        features : {'default', dictionary of lists}, optional
            Features to generate. Default is 'default', which uses most of the features in the `signal_features`
            package. Dictionary key-words should be function names from `signal_features`, and the values are
            lists of dictionaries, each corresponding to a set of parameters (key-word arguments) to run the function
            with.
        window_length : float, optional
            Window length in seconds. Default is 3.0s
        step_size : float, int, optional
            Window step size. If a float between 0 and 1.0, taken as the percentage of a window to skip
            between window centers. If an integer greater or equal to 1, taken as the number of samples to skip
            between window centers. Default is 1.0 (windows with no overlap).
        sampling_frequency : float
            Sampling frequency in Hz of the data. Only required if using stand-alone, and not in a Sequential pipeline

        Examples
        --------
        Run the feature extraction with `signal_range` with the default parameters, and permutation entropy with
        2 different sets of parameters.

        >>> features = {
        ...     'signal_range': [{}],
        ...     'permutation_entropy': [{'order': 3, 'delay': 1, 'normalize': True},
        ...                             {'order': 2, 'delay': 1, 'normalize': False}],
        ...     'jerk_metric': [{}]  # no sampling_rate, will be added on initialization
        ... }
        >>> feat_ext = SignalFeatureExtractor(features=features, sampling_rate=50.0)
        """
        super().__init__(**kwargs)
        # check sign of input values
        super()._check_sign(window_length, 'window_length', pos=True, inc_zero=False)
        super()._check_sign(step_size, 'step_size', pos=True, inc_zero=False)

        self.window_length = window_length  # save for str/repr
        self.step_size = step_size  # save for str/repr

        self.win_l = int(window_length * self.fs)
        if isinstance(step_size, float):
            if step_size == 1.0:
                self.step = self.win_l
            elif 0.0 <= step_size < 1.0:
                tmp_ = int(self.win_l * step_size)
                self.step = tmp_ if tmp_ >= 1 else 1
            else:
                raise ValueError('Float step_size must be between 0.0 and 1.0')
        elif isinstance(step_size, int):
            if step_size >= 1:
                self.step = step_size
            else:
                raise ValueError('Integer step_size must be greater than or equal to 1')
        else:
            raise ValueError('step_size must be either a float or integer')

        # ensure correct feature format
        if features == 'default':
            self.features = SignalFeatureExtractor._check_features(DEFAULT_FEATURES)
        else:
            self.features = SignalFeatureExtractor._check_features(features)

        # add sampling rate to the key-word arguments where necessary
        SignalFeatureExtractor._add_sampling_rate(self.fs, self.features)

        self.n_features = SignalFeatureExtractor._get_n_feats(self.features)

    def _call(self):
        if 'Processed' in self.data:
            days = [i for i in self.data['Processed']['Gait'] if 'Day' in i]
        else:
            days = ['Day 1']

        for iday, day in enumerate(days):
            try:
                start, stop = self.data['Processed']['Gait'][day]['Indices']
            except KeyError:
                start, stop = 0, self.data['Sensors']['Lumbar']['Accelerometer'].shape[0]

            wind_acc = SignalFeatureExtractor._get_windowed_view(
                self.data['Sensors']['Lumbar']['Accelerometer'][start:stop],
                self.win_l,
                self.step,
                ensure_c_contiguity=True  # convert to c-contiguous if not already
            )

            # shape of the expected resulting features
            n_ax = (1 if wind_acc.ndim == 2 else wind_acc.shape[-1])
            fshape = (
                wind_acc.shape[0],
                self.n_features * n_ax
            )
            feats = zeros(fshape)

            # compute the features
            cnt = 0
            for i, fname in enumerate(self.features):
                func = getattr(SF, fname)  # get the function from signal_features
                for kwd in self.features[fname]:  # iterate over the keyword arguments
                    tmp = func(wind_acc, **kwd)

                    if isinstance(tmp, tuple):
                        for ft in tmp:
                            i2 = cnt + n_ax
                            feats[:, cnt:i2] = ft

                            cnt += n_ax
                    else:
                        i2 = cnt + (SF.FEATS_PER_FINC[fname] * n_ax)
                        feats[:, cnt:i2] = tmp
                        cnt += (SF.FEATS_PER_FINC[fname] * n_ax)

            self.data = (f'Processed/Gait/{day}/Signal Features', feats)

    @staticmethod
    def _get_windowed_view(x, window_length, step_size, ensure_c_contiguity=False):
        """
        Return a moving window view over the data

        Parameters
        ----------
        x : numpy.ndarray
            1- or 2-D array of signals to window. Windows occur along the 0 axis. Must be C-contiguous.
        window_length : int
            Window length/size.
        step_size : int
            Step/stride size for windows - how many samples to step from window center to window center.
        ensure_c_contiguity : bool, optional
            Create a new array with C-contiguity if the passed array is not C-contiguous. This *may* result in the
            memory requirements significantly increasing. Default is False, which will raise a ValueError if `x` is
            not C-contiguous

        Returns
        -------
        x_win : numpy.ndarray
            2- or 3-D array of windows of the original data, of shape (..., L[, ...])
        """
        if not (x.ndim in [1, 2]):
            raise ValueError('Array cannot have more than 2 dimensions.')

        if ensure_c_contiguity:
            x = require(x, requirements=['C'])
        else:
            if not x.flags['C_CONTIGUOUS']:
                raise ValueError("Input array must be C-contiguous.  See numpy.ascontiguousarray")

        if x.ndim == 1:
            nrows = ((x.size - window_length) // step_size) + 1
            n = x.strides[0]
            return as_strided(x, shape=(nrows, window_length), strides=(step_size * n, n), writeable=False)

        else:
            k = x.shape[1]
            nrows = ((x.shape[0] - window_length) // step_size) + 1
            n = x.strides[1]

            new_shape = (nrows, window_length, k)
            new_strides = (step_size * k * n, k * n, n)
            return as_strided(x, shape=new_shape, strides=new_strides, writeable=False)

    @staticmethod
    def _check_features(feats):
        """
        Check features to ensure correct input
        """
        if not isinstance(feats, dict):
            raise ValueError("'features' must be a dictionary")
        else:
            for func in feats:
                if getattr(SF, func, None) is None:
                    raise ValueError(f"Function ({func}) not found in signal_features.")
                if not isinstance(feats[func], list):
                    raise ValueError(f"'features' Function ({func}) values must be a list.")

            return deepcopy(feats)

    @staticmethod
    def _add_sampling_rate(fs, feats):
        """
        Add sampling rate key-word argument to any functions that require it
        """
        for func in REQUIRE_SAMPLING_RATE:
            if func in feats:
                for kwd in feats[func]:
                    kwd.update(dict(sampling_rate=fs))

    @staticmethod
    def _get_n_feats(feats, default_feats_per_func=5):
        """
        Get the maximum numer of expected features
        """
        FPF = SF.FEATS_PER_FUNC

        n_feat = 0
        for func in feats:
            n_feat += (FPF[func] if (FPF[func] > 0) else default_feats_per_func) * len(feats[func])

        return n_feat


class WindowData(_BaseProcess):
    def __str__(self):
        return f"Window Data (window_length={self.window_length}, step_size={self.step_size})"

    def __repr__(self):
        return f"WindowData ({self.window_length}, {self.step_size}, {self.ensure_c})"

    def __init__(self, window_length, step_size, **kwargs):
        """
        Window inertial sensor data.

        Parameters
        ----------
        window_length : int
            Window length in samples.
        step_size : int
            Step/stride size for windows - how many samples to step from window center to window center.
        """
        super().__init__(**kwargs)
        # check sign of input values
        super()._check_sign(window_length, 'window_length', pos=True, inc_zero=False)
        super()._check_sign(step_size, 'step_size', pos=True, inc_zero=False)

        self.window_length = window_length
        self.step_size = step_size

    def _call(self):
        if 'Processed' in self.data:
            days = [i for i in self.data['Processed']['Gait'] if 'Day' in i]
        else:
            days = ['Day 1']

        for iday, day in enumerate(days):
            try:
                start, stop = self.data['Processed']['Gait'][day]['Indices']
            except KeyError:
                start, stop = 0, self.data['Sensors']['Lumbar']['Accelerometer'].shape[0]


