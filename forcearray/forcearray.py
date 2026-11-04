"""Read FORCE Level-2 datacube tiles directly into numpy arrays.

Performance notes
-----------------
* QAI images are never stacked. Each QAI is read, turned into a boolean
  "valid" mask with a 65536-entry lookup table (much faster than ``np.isin``)
  and dropped right away.
* A BOA image is only read if its QAI has at least one valid pixel
  (in the current window), so fully cloudy scenes cost one QAI read.
* ``get_data_mean`` and ``get_data_te`` stream over the images and keep
  running sums. They never hold the time series in memory, so memory use is
  about one output array, whatever the number of images.
* ``get_data_stm`` works on row blocks sized to ``max_memory_mb``. Each
  block holds the int16 stack of every image for those rows, and the
  quantiles come from a multi-threaded numba kernel that uses no temporary
  copies. It keeps each pixel's series as int16 and sorts short series
  (up to 128 valid observations, the usual case) with an insertion sort,
  which is much faster than ``np.sort`` on short arrays.
* Files are read in parallel by a thread pool (GDAL releases the GIL while
  decoding).
* Landsat (1000 x 1000) and Sentinel-2 (3000 x 3000) images can be mixed by
  setting ``tile_size``; they are resampled on the fly as they are read.
* ``topleft_xy`` / ``bottomright_xy`` restrict every read to a pixel block
  of the tile, so only that part of each file is decoded.
* ``pixels`` reads any number of single pixels in one pass: each file is
  opened once per image and each internal GeoTIFF block that holds pixels is
  decoded once, however many pixels it holds.
* GDAL is told not to list the tile folder when it opens a file
  (``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``). A FORCE tile folder holds
  thousands of files, and that listing is a large share of the cost of a
  small read.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from itertools import product
import os
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from affine import Affine
from numba import njit, prange

from . import config


def _amul(a, b):
    """``a @ b`` for Affine transforms (or a transform and a point).

    affine 3 deprecates ``*`` in favour of ``@``; older versions only have ``*``.
    """
    try:
        return a @ b
    except TypeError:
        return a * b


VALID_MIN = 0
VALID_MAX = 10000


def toRaster(arr_in, base_raster, out_raster, transform=None, crs=None):
    """Write an array of shape (bands, rows, cols) as an int16 GeoTIFF.

    ``base_raster`` gives the CRS and transform. When the array covers only a
    block of the tile (``topleft_xy`` / ``bottomright_xy``), pass
    ``transform=fa.transform`` (and optionally ``crs=fa.crs``) so the output
    is georeferenced to the block; ``base_raster`` can then be None.
    """
    arr_in = np.asarray(arr_in)
    if arr_in.ndim == 2:
        arr_in = arr_in[np.newaxis]
    band_crs, band_tranform = crs, transform
    if band_crs is None or band_tranform is None:
        with rasterio.open(base_raster) as src:
            band_crs = src.crs if band_crs is None else band_crs
            band_tranform = src.transform if band_tranform is None else band_tranform

    meta = {
        "driver": "GTiff",
        "height": arr_in.shape[1],
        "width": arr_in.shape[2],
        "count": arr_in.shape[0],
        "dtype": np.int16,
        "crs": band_crs,
        "transform": band_tranform,
        "nodata": -9999,
    }

    with rasterio.open(out_raster, "w", **meta) as dst:
        dst.write(arr_in.astype(np.int16, copy=False))


# --------------------------------------------------------------------------- #
# Numba kernel for per-pixel quantiles
# --------------------------------------------------------------------------- #
# Below this many valid observations a pixel's series is sorted with an
# insertion sort, which beats np.sort's call and allocation overhead on short
# arrays. Typical FORCE time series (tens of clear observations per pixel)
# fall below it.
_INSERTION_SORT_MAX = 128


@njit(inline='always')
def _insertion_sort(a, n):
    """Sort a[:n] in place."""
    for i in range(1, n):
        v = a[i]
        j = i - 1
        while j >= 0 and a[j] > v:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v


@njit(parallel=True, cache=True)
def _quantile_kernel(stack, q, nodata, fill, out):
    """Per-pixel linear quantiles along axis 0, ignoring ``nodata``.

    stack : (T, N) int16, contiguous
    q     : (P,) float64 in [0, 1]
    fill  : value written where a pixel has no valid value
    out   : (P, N) float32
    Same method as numpy's default 'linear' interpolation.
    """
    T, N = stack.shape
    P = q.shape[0]
    chunk = 256
    n_chunks = (N + chunk - 1) // chunk
    for c in prange(n_chunks):
        start = c * chunk
        stop = min(start + chunk, N)
        width = stop - start
        # Copy (T, width) into a local buffer so the reads from the
        # (T, N) array are contiguous. Values stay int16 (exact, and half
        # the memory traffic of float64).
        local = np.empty((width, T), dtype=np.int16)
        counts = np.zeros(width, dtype=np.int64)
        for t in range(T):
            row = stack[t]
            for j in range(width):
                v = row[start + j]
                if v != nodata:
                    local[j, counts[j]] = v
                    counts[j] += 1
        for j in range(width):
            n = counts[j]
            if n == 0:
                for k in range(P):
                    out[k, start + j] = fill
                continue
            if n <= _INSERTION_SORT_MAX:
                data = local[j]
                _insertion_sort(data, n)
            else:
                data = np.sort(local[j, :n])
            if n == 1:
                for k in range(P):
                    out[k, start + j] = data[0]
                continue
            for k in range(P):
                pos = q[k] * (n - 1)
                i = int(np.floor(pos))
                g = pos - i
                if i >= n - 1:
                    out[k, start + j] = data[n - 1]
                else:
                    out[k, start + j] = (1.0 - g) * data[i] + g * data[i + 1]


class forcearray():
    def __init__(self, level2_dir, tile, start_date, end_date, sensors, bandlist,
                 best_cso=False, tile_size=None, resampling='nearest',
                 n_workers=None, max_memory_mb=4096,
                 topleft_xy=None, bottomright_xy=None, pixels=None,
                 gdal_options=None, target_size=None, points=None):
        """
        Parameters
        ----------
        level2_dir, tile : str
            The tile folder is ``level2_dir/tile``.
        start_date, end_date : str
            'YYYYMMDD', both inclusive.
        sensors : str
            'all' or a comma-separated list such as 'LND08,LND09,SEN2A'.
        bandlist : list of str
            For example ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'].
        best_cso : bool
            Use the stricter QAI filter.
        tile_size : None, int or (rows, cols)
            Output grid size. ``None`` keeps the native size, which works
            only when every image has the same size. Set it, for example to
            1000 or 3000, to mix Landsat and Sentinel-2. Images are then
            resampled as they are read.
        resampling : str
            Rasterio resampling method used for BOA when an image is resampled
            ('nearest', 'average', 'bilinear', ...). QAI always uses
            'nearest'.
        n_workers : int or None
            Number of threads used to read files. Default: CPU count.
        max_memory_mb : int
            Approximate memory budget for buffers (blocks in ``get_data_stm``
            and images read ahead in the streaming methods). Output arrays
            are not counted.
        topleft_xy, bottomright_xy : None or (X, Y)
            Read only a block of the tile. X is the column and Y the row, in
            pixels of the output grid (the native size, or ``tile_size``
            if set). ``topleft_xy`` is inclusive, ``bottomright_xy`` is
            exclusive, so ``topleft_xy=(0, 0), bottomright_xy=(10, 10)``
            gives a 10 x 10 block. Defaults: (0, 0) and the bottom-right
            corner of the tile, i.e. the whole tile.
        pixels : None or sequence of (X, Y)
            Read only these pixels (same coordinates as ``topleft_xy``).
            Faster than a 1 x 1 block per pixel because every file is read
            once for all pixels. The row axis is dropped from the results:
            ``get_data_mean`` -> (n_pixels, bands), ``get_data_te`` ->
            (n_intervals, n_pixels, bands), ``get_data_stm`` ->
            (n_quantiles, n_pixels, bands). Cannot be combined with
            ``topleft_xy`` / ``bottomright_xy``.
        gdal_options : None or dict
            GDAL configuration options set while reading. Default:
            ``{'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR'}``, which skips
            listing the tile folder each time a file is opened (side-car
            files such as .aux.xml or .ovr are then ignored). Pass ``{}`` to
            use GDAL's defaults.
        target_size
            Deprecated name of ``tile_size``.
        points
            Deprecated name of ``pixels``.

        Attributes
        ----------
        shape : (rows, cols)
            Size of the arrays returned (the block size). ``(1, n_pixels)``
            internally with ``pixels``.
        grid_shape : (rows, cols)
            Size of the whole tile on the output grid.
        transform, crs
            Georeferencing of the block (None if no image was found, or with
            ``pixels``).
        pixels_xy : (n_pixels, 2) int array or None
            Pixel coordinates (X, Y) of ``pixels``.
        pixels_coords : (n_pixels, 2) float array or None
            Map coordinates (x, y in ``crs``) of the centre of each pixel.
        """
        if target_size is not None:
            if tile_size is not None:
                raise TypeError('Pass tile_size only (target_size is its deprecated name).')
            warnings.warn('target_size is deprecated, use tile_size instead.',
                          DeprecationWarning, stacklevel=2)
            tile_size = target_size
        if points is not None:
            if pixels is not None:
                raise TypeError('Pass pixels only (points is its deprecated name).')
            warnings.warn('points is deprecated, use pixels instead.',
                          DeprecationWarning, stacklevel=2)
            pixels = points
        self.tile_dir = os.path.join(level2_dir, tile)
        self.start_date = start_date
        self.end_date = end_date
        self.sensors = sensors
        self.best_cso = best_cso
        self.bandlist = list(bandlist)
        self.n_workers = n_workers or os.cpu_count() or 4
        self.max_memory_mb = max_memory_mb
        self.gdal_options = {'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR'} \
            if gdal_options is None else dict(gdal_options)
        self.resampling = getattr(Resampling, resampling) if isinstance(resampling, str) else resampling
        self.LS_allowed_sensors = ['LND04', 'LND05', 'LND07', 'LND08', 'LND09']
        self.S2_allowed_sensors = ['SEN2A', 'SEN2B', 'SEN2C']

        self.LS_bandlist = dict(config.LS_BANDLIST)
        self.S2_bandlist = dict(config.S2_BANDLIST)

        # Band indices are only computed for sensor families that are
        # actually used, so S2-only bands do not fail on S2-only queries.
        self.LS_band_index = [self.LS_bandlist[b] for b in self.bandlist if b in self.LS_bandlist]
        self.S2_band_index = [self.S2_bandlist[b] for b in self.bandlist if b in self.S2_bandlist]

        # Get list boa, qai
        self.boa_files, self.qai_files = self.__get_image_list()

        # Get sensor list
        self.sensor_list = [s.split('_')[2] for s in self.boa_files]
        self._band_index = [self.__band_index_for(s) for s in self.sensor_list]

        # Get cso value
        self.cso_list = self.__get_cso_value()
        self._qai_lut = np.zeros(65536, dtype=bool)
        self._qai_lut[np.asarray(self.cso_list, dtype=np.int64)] = True

        # Output grid and block (or pixels)
        (self.grid_shape, self._row_align, grid_transform, self.crs,
         self._grid_block) = self.__resolve_grid(tile_size)
        self.pixels_xy = self.pixels_coords = None
        self._pixel_groups = None
        if pixels is not None:
            if topleft_xy is not None or bottomright_xy is not None:
                raise ValueError('Use either pixels or topleft_xy/bottomright_xy, not both.')
            self.pixels_xy = self.__resolve_pixels(pixels)
            self._pixel_groups = self.__group_pixels(self.pixels_xy)
            self._offset, self.shape = (0, 0), (1, len(self.pixels_xy))
            self._row_align = None
            self.transform = None
            if grid_transform is not None:
                xs, ys = _amul(grid_transform, (self.pixels_xy[:, 0] + 0.5, self.pixels_xy[:, 1] + 0.5))
                self.pixels_coords = np.column_stack([xs, ys])
        else:
            self._offset, self.shape = self.__resolve_block(topleft_xy, bottomright_xy)
            x0, y0 = self._offset
            self.transform = None if grid_transform is None \
                else _amul(grid_transform, Affine.translation(x0, y0))
            # Blocks of get_data_stm only line up with the GeoTIFF internal
            # blocks if the block starts on one of their boundaries.
            if self._row_align and y0 % self._row_align:
                self._row_align = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_data_te(self, time_step, nodata=-9999, toInt16=False, trackProgress=False):
        """Temporal mean over consecutive ``time_step``-day intervals.

        Returns an array of shape (n_intervals, rows, cols, bands). Intervals
        start at ``start_date`` and are ``time_step`` days long (Feb 29 is
        not counted). The last interval ends at ``end_date`` and can be
        shorter. ``get_te_dates`` gives the dates of each interval.
        """
        with rasterio.Env(**self.gdal_options):
            return self._squeeze_pixels(
                self._get_data_te(time_step, nodata, toInt16, trackProgress))

    def get_te_dates(self, time_step):
        """First and last day (both inclusive) of each ``get_data_te`` interval.

        Returns two ``datetime64[D]`` arrays of length n_intervals.
        """
        b = self.__generate_date_list(time_step).astype('datetime64[D]')
        return b[:-1], b[1:] - np.timedelta64(1, 'D')

    def get_data_stm(self, p_list, nodata=-9999, toInt16=False, trackProgress=False):
        """Per-pixel quantiles (spectral-temporal metrics).

        Returns an array of shape (len(p_list), rows, cols, bands), or
        (rows, cols, bands) if ``p_list`` is a scalar.
        """
        with rasterio.Env(**self.gdal_options):
            return self._squeeze_pixels(
                self._get_data_stm(p_list, nodata, toInt16, trackProgress))

    def get_data_mean(self, nodata=-9999, toInt16=False):
        """Per-pixel mean of all valid observations, shape (rows, cols, bands)."""
        with rasterio.Env(**self.gdal_options):
            return self._squeeze_pixels(self._get_data_mean(nodata, toInt16))

    def get_data_raw(self, nodata=-9999, drop_empty=False, trackProgress=False):
        """Stack of every image, masked with its QAI.

        Returns ``(raw_stack, date_list, sensor_list)``:

        raw_stack : int16 array of shape (n_images, rows, cols, bands)
            (``(n_images, n_pixels, bands)`` with ``pixels``). Pixels that are
            not valid in the QAI (or outside 0..10000) are set to ``nodata``.
        date_list : list of str
            Acquisition date 'YYYYMMDD' of each image.
        sensor_list : list of str
            Sensor of each image, e.g. 'LND08' or 'SEN2A'.

        Images are sorted by date. With ``drop_empty=True``, images without
        any valid pixel (in the block / pixels read) are left out.

        The whole stack is held in memory: about
        n_images * rows * cols * bands * 2 bytes.
        """
        with rasterio.Env(**self.gdal_options):
            stack, keep = self._get_data_raw(nodata, trackProgress)
        if drop_empty:
            stack = stack[keep]
            idx = np.flatnonzero(keep)
        else:
            idx = range(len(self.boa_files))
        date_list = [self.boa_files[i][:8] for i in idx]
        sensor_list = [self.sensor_list[i] for i in idx]
        return self._squeeze_pixels(stack), date_list, sensor_list

    # Deprecated attribute names (``points`` was renamed to ``pixels``).
    @property
    def points_xy(self):
        warnings.warn('points_xy is deprecated, use pixels_xy instead.',
                      DeprecationWarning, stacklevel=2)
        return self.pixels_xy

    @property
    def points_coords(self):
        warnings.warn('points_coords is deprecated, use pixels_coords instead.',
                      DeprecationWarning, stacklevel=2)
        return self.pixels_coords

    def _squeeze_pixels(self, arr):
        """With ``pixels``, drop the (size 1) row axis: (..., 1, n, B) -> (..., n, B)."""
        return arr if self._pixel_groups is None else arr[..., 0, :, :]

    # ------------------------------------------------------------------ #
    # Computation
    # ------------------------------------------------------------------ #
    def _get_data_te(self, time_step, nodata, toInt16, trackProgress):
        H, W = self.shape
        B = len(self.bandlist)
        target_dates = self.__generate_date_list(time_step)
        n_int = len(target_dates) - 1
        out_dtype = np.int16 if toInt16 else np.float32
        te = np.full((max(n_int, 0), H, W, B), nodata, dtype=out_dtype)
        if n_int <= 0 or not self.boa_files:
            return te

        data_dates = np.array([datetime.strptime(x[:8], "%Y%m%d") for x in self.qai_files],
                              dtype='datetime64[us]')
        interval = np.searchsorted(target_dates.astype('datetime64[us]'), data_dates, side='right') - 1
        order = [i for i in np.argsort(interval, kind='stable') if 0 <= interval[i] < n_int]

        sum_ = np.zeros((B, H, W), dtype=np.int32)
        cnt = np.zeros((H, W), dtype=np.int32)
        current = None
        done = 0
        for idx, (boa, valid) in zip(order, self._stream_images(order)):
            k = interval[idx]
            if current is not None and k != current:
                self._finalize_mean(sum_, cnt, te[current], nodata)
                sum_.fill(0)
                cnt.fill(0)
            current = k
            if boa is not None:
                self._accumulate(sum_, cnt, boa, valid)
            done += 1
            if trackProgress:
                print(f'\rEncoding {done}/{len(order)}', end='', flush=True)
        if current is not None:
            self._finalize_mean(sum_, cnt, te[current], nodata)
        if trackProgress:
            print()
        return te

    def _get_data_stm(self, p_list, nodata, toInt16, trackProgress):
        H, W = self.shape
        B = len(self.bandlist)
        T = len(self.boa_files)
        scalar_q = np.isscalar(p_list)
        q = np.atleast_1d(np.asarray(p_list, dtype=np.float64))
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Quantile value must be between 0 and 1")
        P = len(q)
        out_dtype = np.int16 if toInt16 else np.float32
        stm = np.full((P, H, W, B), nodata, dtype=out_dtype)

        if T > 0:
            # Memory per output row: int16 stack + float32 kernel result
            # + the kernel's per-thread buffers (small).
            per_row = T * B * W * 2 + P * B * W * 4
            rows = self._block_rows(per_row)
            n_blocks = (H + rows - 1) // rows
            stack_nodata = np.int16(-32768)  # internal marker, outside valid range
            with ThreadPoolExecutor(self.n_workers) as ex:
                for bi, r0 in enumerate(range(0, H, rows)):
                    r1 = min(r0 + rows, H)
                    h = r1 - r0
                    stack = np.empty((T, B, h, W), dtype=np.int16)

                    def fill(t, r0=r0, r1=r1, stack=stack):
                        boa, valid = self._read_masked(t, r0, r1)
                        if boa is None:
                            stack[t] = stack_nodata
                        else:
                            boa[:, ~valid] = stack_nodata
                            stack[t] = boa

                    list(ex.map(fill, range(T)))
                    res = np.empty((P, B * h * W), dtype=np.float32)
                    _quantile_kernel(stack.reshape(T, -1), q, stack_nodata,
                                     np.float32(nodata), res)
                    del stack
                    res = res.reshape(P, B, h, W)
                    # (P, B, h, W) -> (P, h, W, B)
                    stm[:, r0:r1] = np.moveaxis(res, 1, -1)
                    del res
                    if trackProgress:
                        print(f'\rBlock {bi + 1}/{n_blocks}', end='', flush=True)
            if trackProgress:
                print()
        if scalar_q:
            stm = stm[0]
        return stm

    def _get_data_raw(self, nodata, trackProgress):
        H, W = self.shape
        B = len(self.bandlist)
        T = len(self.boa_files)
        stack = np.empty((T, H, W, B), dtype=np.int16)
        keep = np.zeros(T, dtype=bool)
        nodata = np.int16(nodata)

        def fill(t):
            boa, valid = self._read_masked(t)
            if boa is None:
                stack[t] = nodata
                return
            keep[t] = True
            boa[:, ~valid] = nodata
            stack[t] = np.moveaxis(boa, 0, -1)  # (B, H, W) -> (H, W, B)

        with ThreadPoolExecutor(self.n_workers) as ex:
            for done, _ in enumerate(ex.map(fill, range(T)), 1):
                if trackProgress:
                    print(f'\rReading {done}/{T}', end='', flush=True)
        if trackProgress and T:
            print()
        return stack, keep

    def _get_data_mean(self, nodata, toInt16):
        H, W = self.shape
        B = len(self.bandlist)
        out_dtype = np.int16 if toInt16 else np.float32
        mean_stack = np.full((H, W, B), nodata, dtype=out_dtype)
        if not self.boa_files:
            return mean_stack
        sum_ = np.zeros((B, H, W), dtype=np.int32)
        cnt = np.zeros((H, W), dtype=np.int32)
        order = list(range(len(self.boa_files)))
        for boa, valid in self._stream_images(order):
            if boa is not None:
                self._accumulate(sum_, cnt, boa, valid)
        self._finalize_mean(sum_, cnt, mean_stack, nodata)
        return mean_stack

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def _read_window(self, path, indexes, r0, r1, resampling):
        """Read rows [r0, r1) of the block, as (bands, r1-r0, cols).

        With ``pixels``, returns (bands, 1, n_pixels).
        """
        with rasterio.open(path) as src:
            if self._pixel_groups is not None:
                out = None
                for gx, gy, gw, gh, sel, rr, cc in self._pixel_groups:
                    arr = self._read_grid(src, indexes, gx, gy, gw, gh, resampling)
                    if out is None:
                        out = np.empty((len(indexes), 1, self.shape[1]), dtype=arr.dtype)
                    out[:, 0, sel] = arr[:, rr, cc]
                return out
            x0, y0 = self._offset
            return self._read_grid(src, indexes, x0, y0 + r0, self.shape[1], r1 - r0, resampling)

    def _read_grid(self, src, indexes, x0, y0, w, h, resampling):
        """Read the window (x0, y0, w, h) of the output grid from ``src``."""
        GH, GW = self.grid_shape
        if src.height == GH and src.width == GW:
            full = x0 == 0 and y0 == 0 and h == GH and w == GW
            return src.read(indexes, window=None if full else Window(x0, y0, w, h))
        # Different size: map the window onto the source pixels and
        # resample while reading.
        sy = src.height / GH
        sx = src.width / GW
        window = Window(x0 * sx, y0 * sy, w * sx, h * sy)
        return src.read(indexes, window=window, out_shape=(len(indexes), h, w),
                        resampling=resampling)

    def _read_masked(self, i, r0=0, r1=None):
        """Read image ``i`` for rows [r0, r1).

        Returns (boa, valid): boa is int16 (bands, rows, cols), or None if
        the window has no valid pixel; valid is a bool (rows, cols) mask.
        """
        if r1 is None:
            r1 = self.shape[0]
        qai = self._read_window(os.path.join(self.tile_dir, self.qai_files[i]), [1], r0, r1,
                                Resampling.nearest)[0]
        valid = self._qai_lut[qai.astype(np.uint16, copy=False)]
        del qai
        if not valid.any():
            return None, valid
        boa = self._read_window(os.path.join(self.tile_dir, self.boa_files[i]),
                                self._band_index[i], r0, r1, self.resampling)
        if boa.dtype == np.int16:
            # A single unsigned comparison checks both 0 <= v and v <= 10000.
            u = boa.view(np.uint16)
            for b in range(u.shape[0]):
                valid &= u[b] <= VALID_MAX
        else:
            for b in range(boa.shape[0]):
                valid &= (boa[b] >= VALID_MIN) & (boa[b] <= VALID_MAX)
            boa = boa.astype(np.int16)
        if not valid.any():
            return None, valid
        return boa, valid

    def _stream_images(self, order):
        """Yield ``_read_masked(i)`` for i in order, reading ahead in a
        thread pool with a limit on how many images are held at once."""
        H, W = self.shape
        per_image = (len(self.bandlist) * 2 + 3) * H * W
        max_inflight = max(1, min(2 * self.n_workers,
                                  int(self.max_memory_mb * 2**20 // per_image)))
        with ThreadPoolExecutor(min(self.n_workers, max_inflight)) as ex:
            pending = deque()
            for i in order:
                pending.append(ex.submit(self._read_masked, i))
                if len(pending) >= max_inflight:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    @staticmethod
    def _accumulate(sum_, cnt, boa, valid):
        boa *= valid          # zero out invalid pixels in place
        sum_ += boa
        cnt += valid

    @staticmethod
    def _finalize_mean(sum_, cnt, out, nodata):
        """Write sum_/cnt into ``out`` (rows, cols, bands); nodata where cnt == 0."""
        empty = cnt == 0
        safe = np.where(empty, 1, cnt)
        for b in range(sum_.shape[0]):
            m = sum_[b] / safe
            m[empty] = nodata
            out[..., b] = m

    def _block_rows(self, per_row_bytes):
        H = self.shape[0]
        rows = int(self.max_memory_mb * 2**20 // max(per_row_bytes, 1))
        rows = max(1, min(H, rows))
        align = self._row_align
        if align and rows >= align:
            rows -= rows % align  # line blocks up with the GeoTIFF internal blocks
        return rows

    def __resolve_grid(self, tile_size):
        shapes = {}
        block_h = {}
        block_w = {}
        transforms = {}
        crs = None
        for i, f in enumerate(self.boa_files):
            fam = 'LS' if self.sensor_list[i] in self.LS_allowed_sensors else 'S2'
            if fam in shapes:
                continue
            with rasterio.open(os.path.join(self.tile_dir, f)) as src:
                shapes[fam] = (src.height, src.width)
                block_h[fam], block_w[fam] = src.block_shapes[0]
                transforms[fam] = src.transform
                crs = src.crs if crs is None else crs

        if tile_size is not None:
            shape = (int(tile_size), int(tile_size)) if np.isscalar(tile_size) \
                else tuple(int(x) for x in tile_size)
        elif not shapes:
            shape = (0, 0)
        elif len(set(shapes.values())) == 1:
            shape = next(iter(shapes.values()))
        else:
            raise ValueError(
                f'Images have different sizes {shapes} (Landsat and Sentinel-2 mixed). '
                'Set tile_size, e.g. tile_size=1000 or tile_size=3000.')

        # Only align blocks when every image is read at its native size.
        align = None
        if shapes and all(s == shape for s in shapes.values()) and len(set(block_h.values())) == 1:
            align = next(iter(block_h.values()))

        # Transform of the output grid: the tile extent split into ``shape``.
        transform = None
        if shapes and shape[0] > 0 and shape[1] > 0:
            fam = next(iter(shapes))
            sh, sw = shapes[fam]
            transform = _amul(transforms[fam], Affine.scale(sw / shape[1], sh / shape[0]))

        # Internal GeoTIFF block size on the output grid (smallest over the
        # sensor families), used to group pixels that are decoded together.
        grid_block = (100, max(shape[1], 1))
        if shapes and shape[0] > 0 and shape[1] > 0:
            grid_block = (
                max(1, int(min(block_h[f] * shape[0] / shapes[f][0] for f in shapes))),
                max(1, int(min(block_w[f] * shape[1] / shapes[f][1] for f in shapes))))
        return shape, align, transform, crs, grid_block

    def __resolve_pixels(self, pixels):
        pts = np.asarray(pixels)
        if pts.ndim == 1 and pts.size == 2:
            pts = pts[np.newaxis]
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            raise ValueError('pixels must be a non-empty sequence of (X, Y) pairs.')
        if not np.all(pts == np.round(pts)):
            raise ValueError('pixels must be integer pixel coordinates (X = column, Y = row).')
        pts = pts.astype(np.int64)
        GH, GW = self.grid_shape
        bad = (pts[:, 0] < 0) | (pts[:, 0] >= GW) | (pts[:, 1] < 0) | (pts[:, 1] >= GH)
        if bad.any() and (GH, GW) != (0, 0):
            raise ValueError(
                f'pixels {pts[bad][:5].tolist()} are outside the tile of {GW} x {GH} pixels '
                f'(need 0 <= X < {GW} and 0 <= Y < {GH}).')
        return pts

    def __group_pixels(self, pts):
        """Group pixels by internal GeoTIFF block, so each block is read once.

        Returns a list of (x0, y0, w, h, sel, rr, cc): the grid window to read,
        the indices of its pixels, and their row/col inside the window.
        """
        bh, bw = self._grid_block
        keys = (pts[:, 1] // bh) * (pts[:, 0].max() // bw + 1) + pts[:, 0] // bw
        groups = []
        for k in np.unique(keys):
            sel = np.flatnonzero(keys == k)
            xs, ys = pts[sel, 0], pts[sel, 1]
            x0, y0 = int(xs.min()), int(ys.min())
            groups.append((x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1,
                           sel, ys - y0, xs - x0))
        return groups

    def __resolve_block(self, topleft_xy, bottomright_xy):
        """Return ((x0, y0), (rows, cols)) of the block to read."""
        GH, GW = self.grid_shape
        if topleft_xy is None and bottomright_xy is None:
            return (0, 0), (GH, GW)
        x0, y0 = (0, 0) if topleft_xy is None else (int(v) for v in topleft_xy)
        x1, y1 = (GW, GH) if bottomright_xy is None else (int(v) for v in bottomright_xy)
        if not self.boa_files and GH == 0 and GW == 0:
            # No image found and no tile_size: nothing to check against.
            return (x0, y0), (max(y1 - y0, 0), max(x1 - x0, 0))
        if not (0 <= x0 < x1 <= GW and 0 <= y0 < y1 <= GH):
            raise ValueError(
                f'Invalid block topleft_xy={(x0, y0)}, bottomright_xy={(x1, y1)} for a '
                f'tile of {GW} x {GH} pixels (X = column, Y = row). Need '
                f'0 <= X_topleft < X_bottomright <= {GW} and '
                f'0 <= Y_topleft < Y_bottomright <= {GH}.')
        return (x0, y0), (y1 - y0, x1 - x0)

    def __band_index_for(self, sensor):
        if sensor in self.LS_allowed_sensors:
            table = self.LS_bandlist
        elif sensor in self.S2_allowed_sensors:
            table = self.S2_bandlist
        else:
            raise ValueError(f'{sensor} does not exist in the allowed list')
        missing = [b for b in self.bandlist if b not in table]
        if missing:
            raise ValueError(f'Band(s) {missing} are not available for sensor {sensor}')
        return [table[b] for b in self.bandlist]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def __generate_date_list(self, time_step):
        start = datetime.strptime(self.start_date, "%Y%m%d")
        end = datetime.strptime(self.end_date, "%Y%m%d")
        date_list = []
        d = start
        noleap_days = 0

        # The first interval always starts at start_date (even on Feb 29).
        date_list.append(d.strftime("%Y%m%d"))

        d += timedelta(days=1)
        while d <= end:
            # Skip Feb 29 completely
            if d.month == 2 and d.day == 29:
                d += timedelta(days=1)
                continue

            noleap_days += 1
            if noleap_days % time_step == 0:
                date_list.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)

        # Close the last interval the day after end_date, so images after the
        # last full interval are kept (in a shorter final interval) instead of
        # being dropped. Intervals are [boundary_k, boundary_k+1).
        if end >= start:
            date_list.append((end + timedelta(days=1)).strftime("%Y%m%d"))

        date_list = np.array(
            [datetime.strptime(date_str, "%Y%m%d") for date_str in date_list],
            dtype=np.datetime64
        )
        return date_list

    def __get_image_list(self):
        start_dt = datetime.strptime(self.start_date, "%Y%m%d")
        end_dt = datetime.strptime(self.end_date, "%Y%m%d")
        boa_files = []
        for filename in os.listdir(self.tile_dir):
            if filename.endswith("BOA.tif"):
                try:
                    file_date_str = filename[:8]  # Extract YYYYMMDD
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")

                    if start_dt <= file_date <= end_dt:
                        boa_files.append(filename)
                except ValueError:
                    continue  # Skip files that don't match the expected pattern
        if self.sensors == 'all':
            pass
        else:
            sensors_list = self.sensors.split(',')
            boa_files = [image for image in boa_files if any(sensor in image for sensor in sensors_list)]
        boa_files.sort()
        qai_files = [image.replace('BOA', 'QAI') for image in boa_files]
        return boa_files, qai_files

    def __get_cso_value(self):
        filtering_list = config.QAI_FILTER_BEST if self.best_cso else config.QAI_FILTER_DEFAULT
        cso_value = [''.join(p) for p in product(*filtering_list.values())]
        cso_value = [x[::-1] for x in cso_value]
        cso_value = [int(x, 2) for x in cso_value]
        cso_value.sort()
        return cso_value
