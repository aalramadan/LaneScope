import pytest

from lanescope.cuda.driver import Driver, DriverError


def test_missing_driver_gives_a_clear_error():
    with pytest.raises(DriverError, match="cannot load the CUDA driver"):
        Driver(library="libcuda-that-does-not-exist.so.1")


def _has_gpu() -> bool:
    try:
        Driver().close()
    except DriverError:
        return False
    return True


@pytest.mark.skipif(not _has_gpu(), reason="a GPU is needed")
def test_download_returns_the_bytes_that_upload_wrote():
    driver = Driver()
    data = bytes(range(256)) * 4
    pointer = driver.alloc(len(data))
    try:
        driver.upload(pointer, data)
        assert driver.download(pointer, len(data)) == data
    finally:
        driver.free(pointer)
        driver.close()
