"""Pre-extraction ID closure; artifact version readers are deliberately unchanged."""

import pytest

from inferref.quantization.dependencies import quantization_dependencies
from inferref.quantization.spec import QuantizationError, QuantizationSpec


def spec(scale, zero=None):
    return QuantizationSpec(logical_shape=(2,), scale_value_id=scale, zero_point_value_id=zero)


def test_shared_parameters_and_selected_parameter_are_not_lost_or_duplicated():
    descriptors = {3: spec(10, 11), 4: spec(10)}
    closure = quantization_dependencies([4, 3, 10, 4], specs=descriptors,
                                       available_ids=[99, 11, 10, 4, 3])
    assert closure.value_ids == (3, 4, 10, 11)
    assert closure.parameter_ids == (10, 11)
    assert closure.quantized_ids == (3, 4)
    assert descriptors == {3: spec(10, 11), 4: spec(10)}
    reordered = quantization_dependencies([3, 4, 10], specs=dict(reversed(list(descriptors.items()))),
                                         available_ids=[3, 4, 10, 11, 99])
    assert reordered == closure


def test_empty_and_plain_projection():
    assert quantization_dependencies([], specs={}, available_ids=[]).value_ids == ()
    plain = quantization_dependencies([3, 1], specs={}, available_ids=[1, 2, 3])
    assert plain.value_ids == (1, 3)
    assert plain.parameter_ids == plain.quantized_ids == ()


@pytest.mark.parametrize("roots,descriptors,available,message", [
    ([1], {}, [], "missing tensor"),
    ([1], {1: spec(2)}, [1], "missing tensor"),
    ([1], {1: spec(2, 3)}, [1, 2], "missing tensor"),
    ([1], {1: spec(1)}, [1], "cycle"),
    ([1], {1: spec(2), 2: spec(1)}, [1, 2], "cycle"),
    ([1], {1: spec(2), 2: spec(3)}, [1, 2, 3], "nested"),
    ([1], {}, [1, 1], "duplicate"),
    ([True], {}, [1], "root value ID"),
    ([1], {}, [True], "available value ID"),
    ([], {True: spec(2)}, [1, 2], "quantized value ID"),
    ([1], {3: spec(2)}, [1, 2], "absent"),
    ([1], {1: {}}, [1], "validated"),
])
def test_invalid_dependencies_fail_before_extraction(roots, descriptors, available, message):
    with pytest.raises(QuantizationError, match=message):
        quantization_dependencies(roots, specs=descriptors, available_ids=available)


def test_unselected_dependencies_are_not_included():
    closure = quantization_dependencies([1], specs={2: spec(99)}, available_ids=[1, 2])
    assert closure.value_ids == (1,)
    assert closure.parameter_ids == ()
    with pytest.raises(QuantizationError, match="missing"):
        quantization_dependencies([1, 2], specs={2: spec(99)}, available_ids=[1, 2])


def test_deep_malformed_chain_is_rejected_without_recursion_error():
    descriptors = {i: spec(i + 1) for i in range(2000)}
    with pytest.raises(QuantizationError, match="nested"):
        quantization_dependencies([0], specs=descriptors, available_ids=range(2001))
    descriptors[1999] = spec(0)
    with pytest.raises(QuantizationError, match="cycle"):
        quantization_dependencies([0], specs=descriptors, available_ids=range(2000))
