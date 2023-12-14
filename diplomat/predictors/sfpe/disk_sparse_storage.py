from collections import deque
from typing import BinaryIO, Any, Dict, Optional, Union, Iterable
from diplomat.predictors.fpe.sparse_storage import (
    ForwardBackwardData,
    AttributeDict,
    ForwardBackwardFrame,
    SettableSequence
)
from diplomat.predictors.sfpe.file_io import DiplomatFPEState, SharedMemory
import multiprocessing


class MonitoredAttributeDict(AttributeDict):
    def __init__(self, backing: DiplomatFPEState):
        self._backing = backing
        super().__init__(backing.get_metadata())

    def __setitem__(self, key: str, val: Any):
        super().__setitem__(key, val)
        self._backing.set_metadata(dict(self.data))


class LIFOCache:
    def __init__(self, size: int):
        self._data = {}
        self._queue = deque()
        self._size = size

    def get(self, index: int, backing: SettableSequence) -> Any:
        if(index in self._data):
            return self._data[index]

        self._data[index] = backing[index]
        self._queue.append(index)

        self._clean_to(self._size, backing)
        return self._data[index]

    def set(self, index: int, backing: SettableSequence, value: Any):
        if(index not in self._data):
            self._queue.append(index)
        self._data[index] = value

        self._clean_to(self._size, backing)

    def _clean_to(self, amount: int, backing: SettableSequence):
        while(len(self._queue) > amount):
            idx = self._queue.popleft()
            backing[idx] = self._data[idx]
            del self._data[idx]

    def flush(self, backing: SettableSequence):
        self._clean_to(0, backing)

    @property
    def size(self) -> int:
        return self._size

    @property
    def __len__(self):
        return len(self._data)

    def __getstate__(self):
        # We don't allow the cache to be pickled to other processes...
        state = self.__dict__
        state["_data"] = {}
        state["_queue"] = deque()
        return state


class IndexIterator:
    def __init__(self, obj: SettableSequence):
        self._obj = obj
        self._idx = 0

    def __next__(self):
        if(self._idx >= len(self._obj)):
            raise StopIteration
        res = self._obj[self._idx]
        self._idx += 1
        return res


class CacheList:
    def __init__(
        self,
        backing: "DiskBackedForwardBackwardData",
        cache: LIFOCache,
        start: int = 0,
        stop: int = None,
        step: int = None
    ):
        self._backing = backing
        self._cache = cache
        stop = stop if(stop is not None) else len(backing)
        step = step if(step is not None) else 1
        self._range = range(start, stop, step)

    def __getitem__(self, index):
        r = self._range[index]
        if(isinstance(r, range)):
            return CacheList(self._backing, self._cache, r.start, r.stop, r.step)
        else:
            return self._cache.get(r, self._backing._frames)

    def __setitem__(self, key, value):
        key = self._range[key]
        if(isinstance(key, range)):
            for i, index in enumerate(key):
                self._cache.set(index, self._backing._frames, value[i])
        else:
            self._cache.set(key, self._backing._frames, value)

    def __len__(self):
        return len(self._range)

    def __iter__(self):
        return IndexIterator(self)


class CacheListContainer:
    def __init__(
        self,
        backing: "DiskBackedForwardBackwardData",
        cache: LIFOCache,
        jump: int,
        start: int = 0,
        stop: Optional[int] = None,
        step: Optional[int] = None
    ):
        self._backing = backing
        self._cache = cache
        self._jump = jump
        self._range = range(
            start,
            stop if(stop is not None) else self._backing_length_chunks(),
            step if(step is not None) else 1
        )

    def __getitem__(self, item: int):
        idx = self._range[item]
        if(isinstance(idx, range)):
            return CacheListContainer(
                self._backing, self._cache, self._jump, idx.start, idx.stop, idx.step
            )
        return CacheList(self._backing, self._cache, self._jump * idx, self._jump * (idx + 1), 1)

    def __setitem__(self, key: int, value: Union[CacheList, Iterable[CacheList]]):
        idx = self._range[key]
        if(isinstance(idx, range)):
            if(isinstance(value, CacheList)):
                value = [value] * len(idx)
            for sub_idx, sub_value in zip(idx, value):
                CacheList(
                    self._backing, self._cache, self._jump * sub_idx, self._jump * (sub_idx + 1), 1
                )[:] = sub_value
        else:
            CacheList(self._backing, self._cache, self._jump * idx, self._jump * (idx + 1), 1)[:] = value

    def _backing_length_chunks(self) -> int:
        return len(self._backing._frames) // self._jump

    def __len__(self):
        return len(self._range)

    def __iter__(self):
        return IndexIterator(self)


class DiskBackedForwardBackwardData(ForwardBackwardData):
    """
    A version of ForwardBackwardData that stores its results on disk instead of
    """
    @classmethod
    def get_shared_memory_size(cls, num_frames: int, num_bps: int):
        return DiplomatFPEState.get_shared_memory_size(num_frames * num_bps)

    def __init__(
        self,
        num_frames: int,
        num_bp: int,
        file_obj: Union[BinaryIO, DiplomatFPEState],
        cache_size: int = 100,
        lock: Optional[multiprocessing.RLock] = None,
        **kwargs
    ):
        """
        Create a new ForwardBackwardData list/object.

        :param num_frames: Number of frames to allocate space for.
        :param num_bp: Number of body parts to allocate space for.
        :param file_obj: The file to use to store files on disk.
        :param cache_size: The size of the cache to use to temporarily store files on disk.
        :param lock: Lock to use when accessing the file (read or write). Defaults to no lock...
        :param kwargs: Additional arguments passed to the storage backend.
        """
        super().__init__(0, 0)
        self._num_bps = num_bp
        self._num_frames = num_frames
        self._cache = LIFOCache(cache_size)
        self.allow_pickle = True

        if(isinstance(file_obj, DiplomatFPEState)):
            self._frames = file_obj
        else:
            self._frames = DiplomatFPEState(file_obj, frame_count=num_frames * num_bp, lock=lock, **kwargs)
        self._metadata = MonitoredAttributeDict(self._frames)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._frames.close()

    def _flush_cache(self):
        # Flush the cache, deleting any new entries...
        self._cache.flush(self._frames)

    def _flush_meta(self):
        self._frames.set_metadata(dict(self._metadata))

    def _flush_all(self):
        self._flush_cache()
        self._flush_meta()
        self._frames.flush()

    def close(self):
        if(self._frames.closed):
            return
        self._flush_cache()
        self._flush_meta()
        self._frames.close()

    @property
    def frames(self) -> SettableSequence[SettableSequence[ForwardBackwardFrame]]:
        """
        Get/Set the frames of this ForwardBackwardData, a 2D list of ForwardBackwardFrame. Indexing is frame, then body
        part.
        """
        return CacheListContainer(
            self,
            self._cache,
            self._num_bps
        )

    @frames.setter
    def frames(self, frames: SettableSequence[SettableSequence[ForwardBackwardFrame]]):
        """
        Frames setter...
        """
        raise NotImplementedError("Not supported for disk backed frame storage.")

    @property
    def num_bodyparts(self) -> int:
        """
        Read-Only: Get the number of body parts stored in this forward backward data object...
        """
        return self._num_bps

    @property
    def num_frames(self) -> int:
        """
        Read-Only: Get the number of frames stored in this forward backward data object...
        """
        return self._num_frames

    @property
    def metadata(self) -> AttributeDict:
        """
        Get/Set the metadata of this forward backward data object. This property can be set to any mapping/dictionary
        type of string to values, but always returns an AttributeDict to allow dot operator access to properties.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, meta: Dict[str, Any]):
        """
        Metadata setter...
        """
        if(self._metadata is meta):
            self._metadata = meta
            return
        self._frames.set_metadata(dict(meta))
        self._metadata = MonitoredAttributeDict(self._frames)

    def copy(self) -> "DiskBackedForwardBackwardData":
        """
        Copy this ForwardBackwardData, returning a new one.
        """
        self._flush_all()
        res = type(self)(self._num_frames, self._num_bps, self._frames, self._cache.size)
        res.allow_pickle = self.allow_pickle
        return res

    def __reduce__(self, *args, **kwargs):
        self._flush_all()
        self.allow_pickle = True
        return super().__reduce__(*args, **kwargs)

    def __del__(self):
        self.close()
