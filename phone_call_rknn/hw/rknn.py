"""ctypes 绑定 librknnrt.so —— 支持 dma-buf 零拷贝输入与 int8 原始输出。

为什么不用 rknnlite：`rknnlite.api.RKNNLite` 只暴露 `inference(numpy)`，
无法做 `rknn_create_mem` / `rknn_set_io_mem(pass_through=1)`，也拿不到
`want_float=0` 的原始 int8 输出。这里直接调 C ABI（结构体布局已用
`sizeof/offsetof` 与板端头文件逐字段核对）。
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

# rknn_query_cmd
QUERY_IN_OUT_NUM = 0
QUERY_INPUT_ATTR = 1
QUERY_OUTPUT_ATTR = 2
QUERY_PERF_RUN = 4
QUERY_SDK_VERSION = 5

# rknn_tensor_type
TENSOR_FLOAT32 = 0
TENSOR_FLOAT16 = 1
TENSOR_INT8 = 2
TENSOR_UINT8 = 3
TENSOR_INT16 = 4
TENSOR_UINT16 = 5
TENSOR_INT32 = 6
TENSOR_UINT32 = 7
TENSOR_INT64 = 8
TENSOR_BOOL = 9
TENSOR_INT4 = 10

# rknn_tensor_format
TENSOR_NCHW = 0
TENSOR_NHWC = 1
TENSOR_NC1HWC2 = 2

# rknn_tensor_qnt_type
QNT_NONE = 0
QNT_DFP = 1
QNT_AFFINE_ASYMMETRIC = 2

# rknn_core_mask
CORE_AUTO = 0
CORE_0 = 1
CORE_1 = 2
CORE_2 = 4
CORE_0_1 = CORE_0 | CORE_1
CORE_0_1_2 = CORE_0_1 | CORE_2

CORE_MASKS = {
    0: CORE_0,
    1: CORE_1,
    2: CORE_2,
    3: CORE_0_1,
    -1: CORE_0_1_2,
    7: CORE_0_1_2,
}


class _TensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class _TensorMem(ctypes.Structure):
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class _Output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class _OutputExtend(ctypes.Structure):
    _fields_ = [
        ("frame_id", ctypes.c_uint64),
    ]


class _RunExtend(ctypes.Structure):
    _fields_ = [
        ("frame_id", ctypes.c_uint64),
        ("non_block", ctypes.c_uint8),
        ("timeout_ms", ctypes.c_int32),
        ("fence_fd", ctypes.c_int32),
    ]


class _InOutNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class _PerfRun(ctypes.Structure):
    _fields_ = [("run_duration", ctypes.c_int64)]


class _SdkVersion(ctypes.Structure):
    _fields_ = [("api_version", ctypes.c_char * 256), ("drv_version", ctypes.c_char * 256)]


def _load_library():
    candidates = [
        os.environ.get("RKNN_RT_LIB"),
        "librknnrt.so",
        "/usr/lib/librknnrt.so",
        "/usr/lib/aarch64-linux-gnu/librknnrt.so",
    ]
    last_error = None
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError as exc:  # pragma: no cover - 依赖板端环境
            last_error = exc
    raise RuntimeError(f"cannot load librknnrt.so: {last_error}")


_lib = _load_library()

_lib.rknn_init.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
_lib.rknn_init.restype = ctypes.c_int
_lib.rknn_destroy.argtypes = [ctypes.c_uint64]
_lib.rknn_destroy.restype = ctypes.c_int
_lib.rknn_query.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
_lib.rknn_query.restype = ctypes.c_int
_lib.rknn_set_core_mask.argtypes = [ctypes.c_uint64, ctypes.c_int]
_lib.rknn_set_core_mask.restype = ctypes.c_int
_lib.rknn_run.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
_lib.rknn_run.restype = ctypes.c_int
_lib.rknn_outputs_get.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(_Output), ctypes.c_void_p]
_lib.rknn_outputs_get.restype = ctypes.c_int
_lib.rknn_outputs_release.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(_Output)]
_lib.rknn_outputs_release.restype = ctypes.c_int
_lib.rknn_create_mem.argtypes = [ctypes.c_uint64, ctypes.c_uint32]
_lib.rknn_create_mem.restype = ctypes.POINTER(_TensorMem)
_lib.rknn_create_mem_from_fd.argtypes = [ctypes.c_uint64, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32]
_lib.rknn_create_mem_from_fd.restype = ctypes.POINTER(_TensorMem)
_lib.rknn_destroy_mem.argtypes = [ctypes.c_uint64, ctypes.POINTER(_TensorMem)]
_lib.rknn_destroy_mem.restype = ctypes.c_int
_lib.rknn_set_io_mem.argtypes = [ctypes.c_uint64, ctypes.POINTER(_TensorMem), ctypes.POINTER(_TensorAttr)]
_lib.rknn_set_io_mem.restype = ctypes.c_int

try:
    _lib.rknn_dup_context.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    _lib.rknn_dup_context.restype = ctypes.c_int
    _HAS_DUP = True
except AttributeError:  # pragma: no cover
    _HAS_DUP = False


@dataclass(frozen=True)
class TensorInfo:
    index: int
    dims: tuple
    fmt: int
    dtype: int
    qnt_type: int
    zp: int
    scale: float
    n_elems: int
    size: int
    w_stride: int
    size_with_stride: int

    @property
    def is_int8(self) -> bool:
        return self.dtype == TENSOR_INT8

    @property
    def is_uint8(self) -> bool:
        return self.dtype == TENSOR_UINT8

    @property
    def quantized(self) -> bool:
        return self.qnt_type == QNT_AFFINE_ASYMMETRIC


def _attr_to_info(attr: _TensorAttr) -> TensorInfo:
    return TensorInfo(
        index=int(attr.index),
        dims=tuple(int(attr.dims[i]) for i in range(int(attr.n_dims))),
        fmt=int(attr.fmt),
        dtype=int(attr.type),
        qnt_type=int(attr.qnt_type),
        zp=int(attr.zp),
        scale=float(attr.scale),
        n_elems=int(attr.n_elems),
        size=int(attr.size),
        w_stride=int(attr.w_stride),
        size_with_stride=int(attr.size_with_stride),
    )


_ERR_TEXT = {
    0: "OK",
    -1: "RKNN_ERR_FAIL",
    -2: "RKNN_ERR_TIMEOUT",
    -3: "RKNN_ERR_DEVICE_UNAVAILABLE",
    -4: "RKNN_ERR_MALLOC_FAIL",
    -5: "RKNN_ERR_PARAM_INVALID",
    -6: "RKNN_ERR_MODEL_INVALID",
    -7: "RKNN_ERR_CTX_INVALID",
    -8: "RKNN_ERR_INPUT_INVALID",
    -9: "RKNN_ERR_OUTPUT_INVALID",
}


def _check(ret: int, what: str) -> int:
    if ret != 0:
        raise RuntimeError(f"{what} failed: ret={ret} ({_ERR_TEXT.get(ret, 'unknown')})")
    return ret


def dequantize(values, zp: int, scale: float, dtype=np.float32):
    """asymmetric affine 反量化：real = scale * (q - zp)。"""
    arr = np.asarray(values)
    out = arr.astype(np.int32)
    out -= int(zp)
    if scale != 1.0:
        out = out * np.float32(scale)
    return out.astype(dtype, copy=False)


class SharedTensor:
    """一块可供 RGA 与多个 RKNN context 共同使用的 dma-buf 张量。"""

    _owned_by_ctx: bool = False

    def __init__(self, owner: "RknnContext", size: int):
        self._owner = owner
        ptr = _lib.rknn_create_mem(owner.ctx, int(size))
        if not ptr:
            raise RuntimeError(f"rknn_create_mem({size}) failed")
        self._ptr = ptr
        self.mem = ptr.contents
        self.size = int(size)
        self.mem_size = int(self.mem.size)
        self.fd = int(self.mem.fd)

    def fd_for(self) -> int:
        return self.fd

    def va(self) -> Optional[np.ndarray]:
        if not self.mem.virt_addr:
            return None
        buf = (ctypes.c_uint8 * self.size).from_address(self.mem.virt_addr)
        return np.ctypeslib.as_array(buf)

    def release(self):
        if self._ptr:
            _lib.rknn_destroy_mem(self._owner.ctx, self._ptr)
            self._ptr = None


def _dtype_itemsize(dtype: int) -> int:
    if dtype in (TENSOR_FLOAT32, TENSOR_INT32, TENSOR_UINT32):
        return 4
    if dtype in (TENSOR_FLOAT16, TENSOR_INT16, TENSOR_UINT16):
        return 2
    if dtype == TENSOR_INT4:
        return 1  # 4bit 不用于 IO
    return 1


class RknnContext:
    """一个 RKNN context，绑定一个模型与一个 NPU core。"""

    def __init__(self, model_path: str | Path, core_mask: Optional[int] = None, name: str = "rknn"):
        self.name = name
        self.model_path = str(model_path)
        self._model = None
        self._ctx = ctypes.c_uint64(0)
        self._input_mem = None       # 自己拥有的输入内存
        self._imported_mem = None    # 从外部 fd 导入的内存
        self._bound = False
        self._input_bytes = 0
        self._outputs = None
        self._output_infos = ()
        self._perf = 0.0
        self._load()
        if core_mask is not None:
            _check(_lib.rknn_set_core_mask(self._ctx, int(core_mask)), f"{name}: set_core_mask")

    # ---- 生命周期 -------------------------------------------------
    def _load(self):
        with open(self.model_path, "rb") as fh:
            data = fh.read()
        self._model = ctypes.create_string_buffer(data, len(data))
        _check(
            _lib.rknn_init(ctypes.byref(self._ctx), self._model, len(data), 0, None),
            f"{self.name}: rknn_init({self.model_path})",
        )
        num = _InOutNum()
        _check(_lib.rknn_query(self._ctx, QUERY_IN_OUT_NUM, ctypes.byref(num), ctypes.sizeof(num)), f"{self.name}: query io num")
        self.n_input = int(num.n_input)
        self.n_output = int(num.n_output)
        if self.n_input <= 0:
            raise RuntimeError(f"{self.name}: model has no input")

        attrs = []
        for i in range(self.n_output):
            attr = _TensorAttr()
            attr.index = i
            _check(
                _lib.rknn_query(self._ctx, QUERY_OUTPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr)),
                f"{self.name}: query output attr {i}",
            )
            attrs.append(attr)
        self._output_attrs = attrs
        self._output_infos = tuple(_attr_to_info(a) for a in attrs)

        in_attr = _TensorAttr()
        in_attr.index = 0
        _check(
            _lib.rknn_query(self._ctx, QUERY_INPUT_ATTR, ctypes.byref(in_attr), ctypes.sizeof(in_attr)),
            f"{self.name}: query input attr",
        )
        self._input_attr = in_attr
        self.input = _attr_to_info(in_attr)

    def sdk_version(self) -> tuple:
        info = _SdkVersion()
        _check(_lib.rknn_query(self._ctx, QUERY_SDK_VERSION, ctypes.byref(info), ctypes.sizeof(info)), "query sdk version")
        return info.api_version.decode(), info.drv_version.decode()

    # ---- 输入绑定 -------------------------------------------------
    def declared_input_size(self, dtype: Optional[int] = None) -> int:
        """按模型输入 shape 计算我们声明给 runtime 的字节数。"""
        dtype = self.input.dtype if dtype is None else dtype
        elems = 1
        for d in self.input.dims:
            elems *= int(d)
        return elems * _dtype_itemsize(dtype)

    def _io_attr(self, pass_through: bool, dtype: Optional[int] = None, fmt: Optional[int] = None) -> _TensorAttr:
        attr = _TensorAttr()
        ctypes.memmove(ctypes.byref(attr), ctypes.byref(self._input_attr), ctypes.sizeof(_TensorAttr))
        attr.pass_through = 1 if pass_through else 0
        if not pass_through:
            # 让 runtime 按我们的内存布局/类型做归一化与 layout 转换（等价 rknn_inputs_set）
            if dtype is not None:
                attr.type = dtype
            if fmt is not None:
                attr.fmt = fmt
            # NHWC 的 w_stride 以像素计；size_with_stride 必须是我们真实提供的字节数
            width = int(self.input.dims[2]) if len(self.input.dims) == 4 else None
            attr.w_stride = width or 0
            attr.h_stride = 0
            attr.size_with_stride = self.declared_input_size(attr.type)
        return attr

    def bind_owned_input(self, pass_through: bool = True, dtype: Optional[int] = None, fmt: Optional[int] = None) -> SharedTensor:
        """分配一块 dma-buf 并绑定为该 context 的输入（RGA 可直接写入）。

        pass_through=False 时由 RKNN runtime 负责 UINT8 NHWC -> 模型输入的
        归一化/类型转换（用于输入是 INT8/FP16 量化张量的模型）。
        """
        if self._bound:
            raise RuntimeError(f"{self.name}: input already bound")
        attr = self._io_attr(pass_through, dtype, fmt)
        size = int(attr.size_with_stride) if not pass_through else int(self.input.size_with_stride)
        mem = _lib.rknn_create_mem(self._ctx, size)
        if not mem:
            raise RuntimeError(f"{self.name}: rknn_create_mem failed")
        _check(_lib.rknn_set_io_mem(self._ctx, mem, ctypes.byref(attr)), f"{self.name}: set_io_mem(input)")
        self._input_mem = mem
        self._input_bytes = int(attr.size_with_stride) if not pass_through else int(self.input.size_with_stride)
        self._bound = True
        tensor = SharedTensor.__new__(SharedTensor)
        tensor._owner = self
        tensor._ptr = None          # 由本 context 释放，不经 SharedTensor
        tensor.mem = mem.contents
        tensor.mem_size = int(mem.contents.size)
        tensor.fd = int(mem.contents.fd)
        tensor.size = int(attr.size_with_stride) if not pass_through else int(self.input.size_with_stride)
        tensor._owned_by_ctx = True
        return tensor

    def bind_shared_input(self, fd: int, size: int, pass_through: bool = True,
                          dtype: Optional[int] = None, fmt: Optional[int] = None,
                          virt_addr: Optional[int] = None) -> None:
        """把外部 dma-buf 绑定为输入（多个 context 共享同一张量，零拷贝）。

        注意：`rknn_create_mem_from_fd` 必须给 `virt_addr`，否则 `set_io_mem`
        会返回 RKNN_ERR_MALLOC_FAIL（2.3.0 实测）。
        """
        if self._bound:
            raise RuntimeError(f"{self.name}: input already bound")
        attr = self._io_attr(pass_through, dtype, fmt)
        if not pass_through:
            size = int(attr.size_with_stride)
        virt = ctypes.c_void_p(virt_addr) if virt_addr else None
        mem = _lib.rknn_create_mem_from_fd(self._ctx, int(fd), virt, int(size), 0)
        if not mem:
            raise RuntimeError(f"{self.name}: rknn_create_mem_from_fd failed")
        _check(_lib.rknn_set_io_mem(self._ctx, mem, ctypes.byref(attr)), f"{self.name}: set_io_mem(shared)")
        self._imported_mem = mem
        self._input_bytes = int(size)
        self._bound = True

    def bind_input_array(self, array: np.ndarray) -> None:
        """非零拷贝兜底：把 numpy 张量拷进已分配的输入内存。"""
        if self._input_mem is None:
            raise RuntimeError(f"{self.name}: no owned input memory")
        tensor = self._input_mem.contents
        raw = np.ctypeslib.as_array((ctypes.c_uint8 * int(self._input_bytes)).from_address(tensor.virt_addr))
        flat = np.ascontiguousarray(array).reshape(-1)
        if flat.size != raw.size:
            raise ValueError(f"{self.name}: input size mismatch {flat.size} != {raw.size}")
        raw[:] = flat

    # ---- 推理 -----------------------------------------------------
    def run(self):
        """执行一次推理，返回 (int8 数组列表, 反量化参数列表)。"""
        if not self._bound:
            raise RuntimeError(f"{self.name}: input not bound")
        _check(_lib.rknn_run(self._ctx, None), f"{self.name}: run")
        outs = (_Output * self.n_output)()
        for i in range(self.n_output):
            outs[i].index = i
            outs[i].want_float = 0
            outs[i].is_prealloc = 0
        _check(_lib.rknn_outputs_get(self._ctx, self.n_output, outs, None), f"{self.name}: outputs_get")
        try:
            arrays = []
            for i in range(self.n_output):
                info = self._output_infos[i]
                nbytes = int(outs[i].size) or info.size
                addr = ctypes.cast(outs[i].buf, ctypes.c_void_p).value
                if addr is None:
                    raise RuntimeError(f"{self.name}: output {i} buffer is NULL")
                if info.is_int8:
                    flat = np.ctypeslib.as_array((ctypes.c_int8 * nbytes).from_address(addr)).copy()
                elif info.is_uint8:
                    flat = np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(addr)).copy()
                elif info.dtype == TENSOR_FLOAT16:
                    flat = np.ctypeslib.as_array((ctypes.c_uint16 * (nbytes // 2)).from_address(addr)).copy().view(np.float16)
                elif info.dtype == TENSOR_FLOAT32:
                    flat = np.ctypeslib.as_array((ctypes.c_float * (nbytes // 4)).from_address(addr)).copy()
                else:
                    raise RuntimeError(f"{self.name}: unsupported output dtype {info.dtype}")
                arrays.append(flat.reshape(info.dims[1:]) if info.dims else flat)
        finally:
            _lib.rknn_outputs_release(self._ctx, self.n_output, outs)
        return arrays

    def query_perf(self) -> float:
        perf = _PerfRun()
        if _lib.rknn_query(self._ctx, QUERY_PERF_RUN, ctypes.byref(perf), ctypes.sizeof(perf)) == 0:
            return perf.run_duration / 1000.0
        return 0.0

    @property
    def ctx(self) -> int:
        """底层 rknn_context 句柄（SharedTensor 等内部结构使用）。"""
        return int(self._ctx.value)

    @property
    def outputs(self) -> tuple:
        return self._output_infos

    def release(self):
        if self._input_mem is not None:
            _lib.rknn_destroy_mem(self._ctx, self._input_mem)
            self._input_mem = None
        if self._imported_mem is not None:
            _lib.rknn_destroy_mem(self._ctx, self._imported_mem)
            self._imported_mem = None
        if self._ctx.value:
            _lib.rknn_destroy(self._ctx)
            self._ctx = ctypes.c_uint64(0)

    def __del__(self):  # pragma: no cover
        try:
            self.release()
        except Exception:
            pass


def warmup(ctx: RknnContext, rounds: int = 5) -> None:
    """用绑定好的张量跑几次，触发 NPU 频率拉高（首次推理明显偏慢）。"""
    for _ in range(rounds):
        ctx.run()
