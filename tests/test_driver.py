import pytest

from lanescope.cuda.driver import Driver, DriverError


def test_missing_driver_gives_a_clear_error():
    with pytest.raises(DriverError, match="cannot load the CUDA driver"):
        Driver(library="libcuda-that-does-not-exist.so.1")
