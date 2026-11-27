import shutil
import subprocess

import pytest

from lanescope import kernel_names

MERGE_SORT = (
    "void cub::CUB_300102_SM_750::detail::merge_sort::DeviceMergeSortBlockSortKernel<"
    "cub::CUB_300102_SM_750::detail::merge_sort::policy_hub<thrust::device_ptr<unsigned long> "
    ">::Policy600, thrust::device_ptr<unsigned long>, cub::NullType*, unsigned int, "
    "elem_less<int> >(thrust::device_ptr<unsigned long>, unsigned int, elem_less<int>)"
)
ELEMENTWISE = (
    "void at::native::(anonymous namespace)::vectorized_elementwise_kernel<4, "
    "at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, "
    "at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)"
)


@pytest.mark.parametrize(
    ("name", "short"),
    [
        ("transpose(float const*, float*, int)", "transpose"),
        ("void kernel<4>(float*)", "kernel<4>"),
        ("histogram", "histogram"),
        (
            "lightning_indexer_kernel_mma<4, 32, 128, 64, GGML_TYPE_F16>",
            "lightning_indexer_kernel_mma<4, 32, 128, 64, GGML_TYPE_F16>",
        ),
        ("(anonymous namespace)::reduce_rows(float*, int)", "reduce_rows"),
        (MERGE_SORT, "DeviceMergeSortBlockSortKernel<…>"),
        (ELEMENTWISE, "vectorized_elementwise_kernel<4, …>"),
    ],
)
def test_short_names_drop_namespaces_return_types_parameters_and_long_arguments(name, short):
    assert kernel_names.short(name) == short


def test_a_short_name_still_too_long_ends_in_an_ellipsis():
    name = "k" * 100
    found = kernel_names.short(name)
    assert len(found) == kernel_names.SHORT_LENGTH
    assert found.endswith(kernel_names.ELLIPSIS)


def test_template_arguments_that_all_go_show_as_one_ellipsis():
    name = "kernel<" + ", ".join(f"ns::Type{index}<int>" for index in range(9)) + ">"
    assert kernel_names.short(name) == "kernel<…>"


@pytest.mark.skipif(shutil.which("c++filt") is None, reason="c++filt is needed")
def test_mangled_names_read_as_their_cpp_names_and_others_stay():
    found = kernel_names.readable(["_Z9transposePKfPfi", "histogram"])
    assert found == {
        "_Z9transposePKfPfi": "transpose(float const*, float*, int)",
        "histogram": "histogram",
    }


def test_mangled_names_stay_as_they_are_without_cpp_filt(monkeypatch):
    monkeypatch.setattr(kernel_names, "_readable", {})
    monkeypatch.setattr(kernel_names.shutil, "which", lambda name: None)
    assert kernel_names.readable(["_Z6kernelILi4EEvPf"]) == {
        "_Z6kernelILi4EEvPf": "_Z6kernelILi4EEvPf"
    }


def test_a_failing_cpp_filt_leaves_the_names_as_they_are(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("c++filt", 30)

    monkeypatch.setattr(kernel_names, "_readable", {})
    monkeypatch.setattr(kernel_names.shutil, "which", lambda name: "/usr/bin/c++filt")
    monkeypatch.setattr(kernel_names.subprocess, "run", fail)
    assert kernel_names.readable(["_Z1kv"]) == {"_Z1kv": "_Z1kv"}


@pytest.mark.parametrize(
    ("name", "template"),
    [
        (
            "void cub::detail::merge_sort::DeviceMergeSortBlockSortKernel<A, B>(C, D)",
            "cub::detail::merge_sort::DeviceMergeSortBlockSortKernel",
        ),
        (MERGE_SORT, "cub::CUB_300102_SM_750::detail::merge_sort::DeviceMergeSortBlockSortKernel"),
        (ELEMENTWISE, "at::native::(anonymous namespace)::vectorized_elementwise_kernel"),
        ("transpose(float const*, float*, int)", "transpose"),
        ("(anonymous namespace)::reduce_rows(float*, int)", "(anonymous namespace)::reduce_rows"),
        ("histogram", "histogram"),
        ("void ns::Outer<int>::kernel<float>(float*)", "ns::Outer::kernel"),
        (
            "void ns::launch<ns::run(int)::{lambda(int)#1}>(int, ns::run(int)::{lambda(int)#1})",
            "ns::launch",
        ),
        ("void ns::apply<&ns::Less::operator<, 4>(int)", "ns::apply"),
        ("void ns::apply<&ns::Shift::operator<<, 4>(int)", "ns::apply"),
        ("void ns::apply<&ns::Call::operator(), 4>(ns::Call)", "ns::apply"),
        ("void ns::apply<&ns::Pointer::operator->, 4>(ns::Pointer)", "ns::apply"),
        ("bool ns::operator< <int>(ns::V<int> const&, ns::V<int> const&)", "ns::operator<"),
        ("void ns::Functor::operator()(int)", "ns::Functor::operator()"),
        ("_Z6kernelILi4EEvPf", "_Z6kernelILi4EEvPf"),
    ],
)
def test_the_template_of_a_kernel_keeps_its_namespaces_and_drops_every_argument_list(
    name, template
):
    assert kernel_names.template_name(name) == template


def test_short_names_keep_the_plain_arguments_after_an_operator_argument():
    types = ", ".join(f"ns::T{index}<int>" for index in range(6))
    name = f"void ns::apply<4, &ns::Cmp::operator<, {types}, 8>(int)"
    assert kernel_names.short(name) == "apply<4, …, 8>"
