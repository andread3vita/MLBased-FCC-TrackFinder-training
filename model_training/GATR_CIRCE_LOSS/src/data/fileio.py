import math
import os
import time
import awkward as ak
import tqdm
from src.data.tools import _concat
from src.logger.logger import _logger
import re
import numpy as np
from functools import lru_cache


_READ_RETRY_DELAYS_SECONDS = (1, 2, 4, 8)

def _read_hdf5(filepath, branches, load_range=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k: getattr(f.root, k)[:] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = v[start:stop]
    return ak.Array(outputs)


def _read_root(filepath, branches, load_range=None, treename=None):
    import uproot
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(
                    'Need to specify `treename` as more than one trees are found in file %s: %s' %
                    (filepath, str(branches)))
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        outputs = tree.arrays(filter_name=branches, entry_start=start, entry_stop=stop)
    return outputs


def _read_awkd(filepath, branches, load_range=None):
    import awkward0
    with awkward0.load(filepath) as f:
        outputs = {k: f[k] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = ak.from_awkward0(v[start:stop])
    return ak.Array(outputs)


@lru_cache(maxsize=None)
def _parquet_layout(filepath):
    """Cache the small footer information needed for row-group pruning."""
    metadata = ak.metadata_from_parquet(filepath)
    columns = frozenset(name.split(".", 1)[0] for name in metadata["columns"])
    return tuple(metadata["col_counts"]), columns


def _read_parquet(filepath, branches, load_range=None):
    """Read only the row groups overlapping ``load_range``.

    ``ak.from_parquet`` is eager. The old implementation read every row group
    and sliced afterwards, which multiplied I/O when ``fetch_step < 1``.
    """
    row_counts, available_columns = _parquet_layout(filepath)
    read_branches = [
        name
        for name in branches
        if name in available_columns or name != "file_number"
    ]

    if load_range is None:
        return ak.from_parquet(filepath, columns=read_branches)

    offsets = np.concatenate(([0], np.cumsum(row_counts, dtype=np.int64)))
    num_rows = int(offsets[-1])
    if num_rows == 0:
        return ak.from_parquet(filepath, columns=read_branches)

    start = min(math.trunc(load_range[0] * num_rows), num_rows)
    stop = min(num_rows, max(start + 1, math.trunc(load_range[1] * num_rows)))
    if start >= stop:
        return ak.from_parquet(filepath, columns=read_branches, row_groups=[])

    first_group = int(np.searchsorted(offsets[1:], start, side="right"))
    last_group = int(np.searchsorted(offsets[1:], stop - 1, side="right")) + 1
    row_groups = list(range(first_group, last_group))
    outputs = ak.from_parquet(
        filepath, columns=read_branches, row_groups=row_groups
    )

    local_start = start - int(offsets[first_group])
    local_stop = stop - int(offsets[first_group])
    return outputs[local_start:local_stop]


def _read_file_with_retries(filepath, branches, load_range=None, treename=None):
    """Read one file strictly, retrying transient failures before giving up.

    There are five total attempts.  Failures before the final attempt wait for
    1, 2, 4, and 8 seconds respectively.  A permanently unreadable or empty
    file is never silently omitted from the dataset: the final RuntimeError is
    chained from the original reader exception so DataLoader reports its cause.
    """
    ext = os.path.splitext(filepath)[1]
    if ext not in ('.h5', '.root', '.awkd', '.parquet'):
        raise RuntimeError(
            'File %s of type `%s` is not supported!' % (filepath, ext)
        )

    max_attempts = len(_READ_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            if ext == '.h5':
                result = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                result = _read_root(
                    filepath,
                    branches,
                    load_range=load_range,
                    treename=treename,
                )
            elif ext == '.awkd':
                result = _read_awkd(filepath, branches, load_range=load_range)
            else:
                result = _read_parquet(
                    filepath, branches, load_range=load_range
                )

            if result is None or len(result) == 0:
                raise RuntimeError(
                    f'Reader returned zero entries for {filepath!r} '
                    f'with `load_range`={load_range}'
                )
            return result
        except Exception as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    f'Failed to read {filepath!r} after {max_attempts} attempts '
                    f'with `load_range`={load_range}'
                ) from error

            delay = _READ_RETRY_DELAYS_SECONDS[attempt - 1]
            _logger.warning(
                'Read attempt %d/%d failed for %s with %s: %s; '
                'retrying in %d second(s)',
                attempt,
                max_attempts,
                filepath,
                type(error).__name__,
                error,
                delay,
            )
            time.sleep(delay)


def _read_files(filelist, branches, load_range=None, show_progressbar=False, **kwargs):
    branches = list(branches)
    table = []
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)
    for filepath in filelist:
        a = _read_file_with_retries(
            filepath,
            branches,
            load_range=load_range,
            treename=kwargs.get('treename', None),
        )

        # New Parquet files store this column. Keep filename inference for
        # legacy ROOT/Parquet inputs so mixed-format validation still works.
        if "file_number" not in a.fields:
            base = os.path.basename(filepath)
            match = re.search(r'(\d+)', base)
            fid = int(match.group(1)) if match else 0
            a = ak.with_field(
                a, np.full(len(a), fid, dtype=np.int32), "file_number"
            )
        table.append(a)
    table = _concat(table)  # ak.Array
    if len(table) == 0:
        raise RuntimeError(f'Zero entries loaded when reading files {filelist} with `load_range`={load_range}.')
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    import uproot
    if compression == -1:
        compression = uproot.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        tree = fout.mktree(treename, {k: v.dtype for k, v in table.items()})
        start = 0
        while start < len(list(table.values())[0]) - 1:
            tree.extend({k: v[start:start + step] for k, v in table.items()})
            start += step
