# Read FORCE L2 data directly from dacube into numpy array

# Install
```
pip install git+https://github.com/vudongpham/ForceArray.git
```


# Usage
```python
from forcearray import forcearray

fa = forcearray('/data/level2', 'X0001_Y0001', '20200101', '20201231',
                sensors='LND08,LND09', bandlist=['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'])

mean = fa.get_data_mean(toInt16=True)                       # (rows, cols, bands)
te   = fa.get_data_te(time_step=16, toInt16=True)           # (intervals, rows, cols, bands)
stm  = fa.get_data_stm([0.1, 0.5, 0.9], toInt16=True)       # (quantiles, rows, cols, bands)
```

Optional arguments of `forcearray(...)`:

| argument | default | meaning |
|---|---|---|
| `target_size` | `None` | Output grid size (e.g. `1000` or `3000`). Needed to mix Landsat (1000 x 1000) with Sentinel-2 (3000 x 3000); images are resampled as they are read. |
| `resampling` | `'nearest'` | BOA resampling when `target_size` differs from an image's size (QAI always uses nearest). |
| `n_workers` | CPU count | Threads used to read files in parallel. |
| `max_memory_mb` | `4096` | Approximate budget for working buffers (the block size in `get_data_stm`, read-ahead in the other methods). Output arrays are not counted. |

# Performance
* QAI images are never stacked: each one becomes a valid-pixel mask through a lookup table and is then dropped.
* BOA is skipped when its QAI has no valid pixel (fully cloudy scenes).
* `get_data_mean` / `get_data_te` keep running sums, so memory does not grow with the number of images.
* `get_data_stm` processes row blocks with an int16 stack and a parallel numba quantile kernel.
* When `toInt16=False`, outputs are `float32`.
