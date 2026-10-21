import ctypes
from collections.abc import Sequence

_ATTRIBUTE_MAJOR = 75
_ATTRIBUTE_MINOR = 76


def version_text(version: int) -> str:
    """Return a CUDA version number of the driver API, such as 12090, as text such as 12.9."""
    return f"{version // 1000}.{version % 1000 // 10}"


class DriverError(RuntimeError):
    """A driver call failed, or no CUDA driver is present."""


class Driver:
    """The primary context of one GPU, with the calls the calibration kernels need."""

    def __init__(self, library: str = "libcuda.so.1", device: int = 0) -> None:
        try:
            # A global load lets the recorder find driver functions by name.
            self._lib = ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise DriverError(
                f"cannot load the CUDA driver {library} ({error}). Run on a machine with an "
                "NVIDIA GPU, such as WSL2 with the Windows driver installed."
            ) from error
        self._call("cuInit", 0)
        self._device = ctypes.c_int()
        self._call("cuDeviceGet", ctypes.byref(self._device), device)
        self._context = ctypes.c_void_p()
        self._call("cuDevicePrimaryCtxRetain", ctypes.byref(self._context), self._device)
        self._call("cuCtxSetCurrent", self._context)

    def name(self) -> str:
        """Return the product name of the GPU."""
        buffer = ctypes.create_string_buffer(256)
        self._call("cuDeviceGetName", buffer, len(buffer), self._device)
        return buffer.value.decode()

    def arch(self) -> str:
        """Return the architecture name of the GPU, such as sm_89."""
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._call("cuDeviceGetAttribute", ctypes.byref(major), _ATTRIBUTE_MAJOR, self._device)
        self._call("cuDeviceGetAttribute", ctypes.byref(minor), _ATTRIBUTE_MINOR, self._device)
        return f"sm_{major.value}{minor.value}"

    def version(self) -> int:
        """Return the CUDA version the driver supports, such as 12090."""
        version = ctypes.c_int()
        self._call("cuDriverGetVersion", ctypes.byref(version))
        return version.value

    def load(self, image: bytes) -> ctypes.c_void_p:
        """Load a cubin and return its module."""
        module = ctypes.c_void_p()
        self._call("cuModuleLoadData", ctypes.byref(module), ctypes.c_char_p(image))
        return module

    def function(self, module: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
        """Return a kernel of a loaded module."""
        function = ctypes.c_void_p()
        self._call("cuModuleGetFunction", ctypes.byref(function), module, name.encode())
        return function

    def alloc(self, size: int) -> int:
        """Allocate device memory and return its address."""
        pointer = ctypes.c_uint64()
        self._call("cuMemAlloc_v2", ctypes.byref(pointer), ctypes.c_size_t(size))
        return pointer.value

    def free(self, pointer: int) -> None:
        """Free device memory."""
        self._call("cuMemFree_v2", ctypes.c_uint64(pointer))

    def upload(self, pointer: int, data: bytes) -> None:
        """Copy bytes to device memory."""
        self._call("cuMemcpyHtoD_v2", ctypes.c_uint64(pointer), data, ctypes.c_size_t(len(data)))

    def launch(
        self,
        function: ctypes.c_void_p,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        arguments: Sequence[ctypes._SimpleCData | ctypes.Array],
    ) -> None:
        """Launch a kernel with its arguments and wait for it to finish."""
        pointers = (ctypes.c_void_p * len(arguments))(
            *[ctypes.cast(ctypes.byref(argument), ctypes.c_void_p) for argument in arguments]
        )
        self._call("cuLaunchKernel", function, *grid, *block, 0, None, pointers, None)
        self._call("cuCtxSynchronize")

    def close(self) -> None:
        """Release the primary context."""
        self._call("cuDevicePrimaryCtxRelease_v2", self._device)

    def _call(self, name: str, *arguments) -> None:
        status = getattr(self._lib, name)(*arguments)
        if status != 0:
            text = ctypes.c_char_p()
            self._lib.cuGetErrorString(status, ctypes.byref(text))
            message = text.value.decode() if text.value else f"error {status}"
            raise DriverError(f"{name} failed: {message}")
