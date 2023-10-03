import pytest
import numpy as np
import random

from core.data_input import Variable, Geometry, Data
from interpolation.interpolation_utils import InterpolationUtils


class DataForTests:
    def __init__(self, length=200) -> None:
        self.length = length
        self.data = [self.create_data_class() for i in range(self.length)]

    def create_data_class(self):
        # create dictionary of variables
        input = {
            "variable_1": np.array(random.sample(range(10, 30), 10)),
            "variable_2": np.array(random.sample(range(10, 30), 10)),
            "depth": np.array(random.sample(range(1, 30), 10)),
        }
        variable_1 = Variable(label="variable_1", value=input["variable_1"])
        variable_2 = Variable(label="variable_2", value=input["variable_2"])
        location = Geometry(
            x=random.sample(range(1, 30), 1)[0], y=random.sample(range(1, 30), 1)[0]
        )
        data = Data(
            location=location,
            variables=[variable_1, variable_2],
            independent_variable=Variable(label="depth", value=input["depth"]),
        )
        return data


class TestInterpolationUtils:
    @pytest.mark.unittest
    def test_get_data_for_interpolation_list_data(self):
        datafortest = DataForTests(20)
        assert len(datafortest.data) == 20
        (
            value_interpolation,
            points_interpolation,
        ) = InterpolationUtils.get_data_for_interpolation(
            datafortest.data,
            value_name="variable_2",
            use_independent_variable_as_z=True,
        )
        assert value_interpolation.shape == (200,)
        assert points_interpolation.shape == (200, 3)

    @pytest.mark.unittest
    def test_get_data_for_interpolation_single_data(self):
        datafortest = DataForTests(1)
        assert len(datafortest.data) == 1
        (
            value_interpolation,
            points_interpolation,
        ) = InterpolationUtils.get_data_for_interpolation(
            datafortest.data[0],
            value_name="variable_2",
            use_independent_variable_as_z=True,
        )
        assert value_interpolation.shape == (10,)
        assert points_interpolation.shape == (10, 3)

    @pytest.mark.unittest
    def test_get_data_for_interpolation_list_data_2d(self):
        datafortest = DataForTests(20)
        assert len(datafortest.data) == 20
        (
            value_interpolation,
            points_interpolation,
        ) = InterpolationUtils.get_data_for_interpolation(
            datafortest.data,
            value_name="variable_2",
            use_independent_variable_as_z=False,
        )
        assert value_interpolation.shape == (200,)
        assert points_interpolation.shape == (200, 2)

    @pytest.mark.unittest
    def test_get_data_for_interpolation_single_data_2d(self):
        datafortest = DataForTests(1)
        assert len(datafortest.data) == 1
        (
            value_interpolation,
            points_interpolation,
        ) = InterpolationUtils.get_data_for_interpolation(
            datafortest.data[0],
            value_name="variable_2",
            use_independent_variable_as_z=False,
        )
        assert value_interpolation.shape == (10,)
        assert points_interpolation.shape == (10, 2)

    @pytest.mark.unittest
    def test_get_points_for_interpolation_list_data(self):
        datafortest = DataForTests(20)
        assert len(datafortest.data) == 20
        points_interpolation = InterpolationUtils.get_points_for_interpolation(
            datafortest.data,
            use_independent_variable_as_z=True,
        )
        assert points_interpolation.shape == (200, 3)

    @pytest.mark.unittest
    def test_get_points_for_interpolation_single_data(self):
        datafortest = DataForTests(1)
        assert len(datafortest.data) == 1
        points_interpolation = InterpolationUtils.get_points_for_interpolation(
            datafortest.data[0],
            use_independent_variable_as_z=True,
        )
        assert points_interpolation.shape == (10, 3)

    @pytest.mark.unittest
    def test_get_points_for_interpolation_list_data_2d(self):
        datafortest = DataForTests(20)
        assert len(datafortest.data) == 20
        points_interpolation = InterpolationUtils.get_points_for_interpolation(
            datafortest.data,
            use_independent_variable_as_z=False,
        )
        assert points_interpolation.shape == (200, 2)

    @pytest.mark.unittest
    def test_get_points_for_interpolation_single_data_2d(self):
        datafortest = DataForTests(1)
        assert len(datafortest.data) == 1
        points_interpolation = InterpolationUtils.get_points_for_interpolation(
            datafortest.data[0],
            use_independent_variable_as_z=False,
        )
        assert points_interpolation.shape == (10, 2)
