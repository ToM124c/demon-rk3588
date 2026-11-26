import ctypes
import threading

import numpy as np

RK_FORMAT_BGR_888 = 0x7 << 8
RK_FORMAT_RGB_888 = 0x2 << 8
RK_FORMAT_YCbCr_420_SP = 0xA << 8
RK_FORMAT_YCbCr_420_SP_10B = 0x20 << 8
IM_STATUS_SUCCESS = 1

NV12_PIXEL_FORMATS = {
    "nv12": RK_FORMAT_YCbCr_420_SP,
    "nv12_10le40": RK_FORMAT_YCbCr_420_SP_10B,
}


class im_rect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]


class im_colorkey_range(ctypes.Structure):
    _fields_ = [("max", ctypes.c_int), ("min", ctypes.c_int)]


class im_nn_t(ctypes.Structure):
    _fields_ = [
        ("scale_r", ctypes.c_int),
        ("scale_g", ctypes.c_int),
        ("scale_b", ctypes.c_int),
        ("offset_r", ctypes.c_int),
        ("offset_g", ctypes.c_int),
        ("offset_b", ctypes.c_int),
    ]


class _AlphaUnion(ctypes.Union):
    _fields_ = [
        ("global_alpha", ctypes.c_int),
        ("alpha0", ctypes.c_uint16),
        ("alpha1", ctypes.c_uint16),
    ]


class rga_buffer_t(ctypes.Structure):
    _fields_ = [
        ("vir_addr", ctypes.c_void_p),
        ("phy_addr", ctypes.c_void_p),
        ("fd", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("wstride", ctypes.c_int),
        ("hstride", ctypes.c_int),
        ("format", ctypes.c_int),
        ("color_space_mode", ctypes.c_int),
        ("alpha_u", _AlphaUnion),
        ("rd_mode", ctypes.c_int),
        ("color", ctypes.c_int),
        ("colorkey_range", im_colorkey_range),
        ("nn", im_nn_t),
        ("rop_code", ctypes.c_int),
        ("handle", ctypes.c_uint32),
    ]


class im_handle_param_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
    ]


class RGAHelper:
    """Minimal ctypes wrapper for Rockchip librga im2d API."""

    def __init__(self, lib_name="librga.so"):
        self._lib = ctypes.CDLL(lib_name)
        self._bind()

    def _bind(self):
        lib = self._lib
        lib.importbuffer_virtualaddr.argtypes = [ctypes.c_void_p, ctypes.POINTER(im_handle_param_t)]
        lib.importbuffer_virtualaddr.restype = ctypes.c_uint32

        lib.wrapbuffer_handle_t.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.wrapbuffer_handle_t.restype = rga_buffer_t

        lib.imresize_t.argtypes = [
            rga_buffer_t,
            rga_buffer_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.imresize_t.restype = ctypes.c_int

        lib.imfill_t.argtypes = [rga_buffer_t, im_rect, ctypes.c_int, ctypes.c_int]
        lib.imfill_t.restype = ctypes.c_int

        lib.imcvtcolor_t.argtypes = [
            rga_buffer_t,
            rga_buffer_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.imcvtcolor_t.restype = ctypes.c_int

        lib.releasebuffer_handle.argtypes = [ctypes.c_uint32]
        lib.releasebuffer_handle.restype = ctypes.c_int

    @staticmethod
    def _check(status, operation):
        if status != IM_STATUS_SUCCESS:
            raise RuntimeError("RGA {} failed with status {}".format(operation, status))

    def import_buffer(self, array, width, height, fmt):
        array = np.ascontiguousarray(array)
        param = im_handle_param_t(width, height, fmt)
        handle = self._lib.importbuffer_virtualaddr(array.ctypes.data, ctypes.byref(param))
        if handle == 0:
            raise RuntimeError("RGA importbuffer_virtualaddr failed")
        return handle, array

    def wrap_handle(self, handle, width, height, wstride, hstride, fmt):
        return self._lib.wrapbuffer_handle_t(handle, width, height, wstride, hstride, fmt)

    def release_handle(self, handle):
        if handle:
            self._lib.releasebuffer_handle(handle)

    def fill(self, dst_buf, color_bgr):
        rect = im_rect(0, 0, dst_buf.width, dst_buf.height)
        color = (color_bgr[2] << 16) | (color_bgr[1] << 8) | color_bgr[0]
        status = self._lib.imfill_t(dst_buf, rect, color, 1)
        self._check(status, "imfill")

    def resize(self, src_buf, dst_buf):
        status = self._lib.imresize_t(src_buf, dst_buf, 0.0, 0.0, 0, 1)
        self._check(status, "imresize")

    def cvtcolor(self, src_buf, dst_buf, src_fmt, dst_fmt):
        status = self._lib.imcvtcolor_t(src_buf, dst_buf, src_fmt, dst_fmt, 0, 1)
        self._check(status, "imcvtcolor")


class RGALetterbox:
    """Thread-local RGA letterbox buffers for multi-threaded inference."""

    def __init__(self, size=640, color=(114, 114, 114)):
        self.size = int(size)
        self.color = tuple(color)
        self._rga = RGAHelper()
        self._local = threading.local()

    def _state(self):
        state = getattr(self._local, "state", None)
        if state is None:
            state = self._create_state()
            self._local.state = state
        return state

    def _create_state(self):
        canvas = np.empty((self.size, self.size, 3), dtype=np.uint8)
        rgb = np.empty((self.size, self.size, 3), dtype=np.uint8)
        canvas_handle, canvas = self._rga.import_buffer(canvas, self.size, self.size, RK_FORMAT_BGR_888)
        rgb_handle, rgb = self._rga.import_buffer(rgb, self.size, self.size, RK_FORMAT_RGB_888)
        canvas_buf = self._rga.wrap_handle(
            canvas_handle, self.size, self.size, self.size, self.size, RK_FORMAT_BGR_888
        )
        rgb_buf = self._rga.wrap_handle(
            rgb_handle, self.size, self.size, self.size, self.size, RK_FORMAT_RGB_888
        )
        return {
            "canvas": canvas,
            "rgb": rgb,
            "bgr_full": None,
            "canvas_handle": canvas_handle,
            "rgb_handle": rgb_handle,
            "canvas_buf": canvas_buf,
            "rgb_buf": rgb_buf,
        }

    def _nv12_rk_format(self, pixel_format):
        return NV12_PIXEL_FORMATS.get(pixel_format, RK_FORMAT_YCbCr_420_SP)

    def nv12_to_bgr(self, nv12_data, width, height, pixel_format="nv12"):
        state = self._state()
        bgr = state.get("bgr_full")
        if bgr is None or bgr.shape[0] != height or bgr.shape[1] != width:
            bgr = np.empty((height, width, 3), dtype=np.uint8)
            state["bgr_full"] = bgr

        src = np.ascontiguousarray(nv12_data)
        src_fmt = self._nv12_rk_format(pixel_format)
        src_handle = None
        bgr_handle = None
        try:
            src_handle, src = self._rga.import_buffer(src, width, height, src_fmt)
            src_buf = self._rga.wrap_handle(src_handle, width, height, width, height, src_fmt)
            bgr_handle, bgr = self._rga.import_buffer(bgr, width, height, RK_FORMAT_BGR_888)
            bgr_buf = self._rga.wrap_handle(bgr_handle, width, height, width, height, RK_FORMAT_BGR_888)
            self._rga.cvtcolor(src_buf, bgr_buf, src_fmt, RK_FORMAT_BGR_888)
        finally:
            self._rga.release_handle(src_handle)
            self._rga.release_handle(bgr_handle)
        state["bgr_full"] = bgr
        return bgr

    def letterbox_from_nv12(self, nv12_data, width, height, pixel_format="nv12"):
        bgr = self.nv12_to_bgr(nv12_data, width, height, pixel_format=pixel_format)
        rgb, scale, pad_x, pad_y = self.letterbox(bgr)
        return rgb, scale, pad_x, pad_y, bgr

    def letterbox(self, image):
        src_h, src_w = image.shape[:2]
        scale = min(self.size / src_w, self.size / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        pad_x = (self.size - new_w) / 2
        pad_y = (self.size - new_h) / 2
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))

        state = self._state()
        src = np.ascontiguousarray(image)
        src_handle = None
        patch_handle = None
        try:
            src_handle, src = self._rga.import_buffer(src, src_w, src_h, RK_FORMAT_BGR_888)
            src_buf = self._rga.wrap_handle(src_handle, src_w, src_h, src_w, src_h, RK_FORMAT_BGR_888)

            self._rga.fill(state["canvas_buf"], self.color)

            patch = state["canvas"][top: top + new_h, left: left + new_w]
            patch_handle, patch = self._rga.import_buffer(patch, new_w, new_h, RK_FORMAT_BGR_888)
            patch_buf = self._rga.wrap_handle(
                patch_handle, new_w, new_h, self.size, new_h, RK_FORMAT_BGR_888
            )
            self._rga.resize(src_buf, patch_buf)
            self._rga.cvtcolor(state["canvas_buf"], state["rgb_buf"], RK_FORMAT_BGR_888, RK_FORMAT_RGB_888)
        finally:
            self._rga.release_handle(src_handle)
            self._rga.release_handle(patch_handle)

        return state["rgb"], scale, float(left), float(top)


_rga_letterbox = None
_rga_init_error = None


def get_rga_letterbox(size=640, color=(114, 114, 114)):
    global _rga_letterbox, _rga_init_error
    if _rga_letterbox is not None:
        return _rga_letterbox
    if _rga_init_error is not None:
        return None
    try:
        _rga_letterbox = RGALetterbox(size=size, color=color)
    except OSError as exc:
        _rga_init_error = exc
        return None
    return _rga_letterbox
