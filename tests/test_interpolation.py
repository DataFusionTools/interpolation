from interpolation.interpolation import (
    InverseDistance,
    NaturalNeighbor,
    CustomKriging,
    VariogramModel,
    Nearest,
)
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt


class TestNearest:
    @pytest.mark.unittest
    def test_nearest(self):
        # define some data to interpolate
        data = DataForTests(length=50)
        # initialize class
        interpolator = Nearest()
        assert interpolator
        # interpolate
        plt.clf()
        plt.plot()
        plt.scatter(data.points_two_d.T[0], data.points_two_d.T[1], c=data.value)
        interpolator.interpolate(data.points_two_d, data.value)
        # create more points for test
        test_data = DataForTests(length=20)
        interpolator.predict(test_data.points_two_d)
        assert len(interpolator.zn) == 20


class DataForTests:
    def __init__(self, length=200) -> None:
        self.length = length
        self.points_two_d = self.create_random_points(two_d=True)
        self.points_three_d = self.create_random_points(two_d=False)
        self.value = self.create_values()

    def create_random_points(self, two_d):
        if two_d:
            return np.array(
                random.sample(
                    [[float(x), float(y)] for x in range(101) for y in range(101)],
                    self.length,
                )
            )
        else:
            return np.array(
                random.sample(
                    [
                        [float(x), float(y), float(z)]
                        for x in range(101)
                        for y in range(101)
                        for z in range(101)
                    ],
                    self.length,
                )
            )

    def create_values(self):
        return np.array(random.sample(range(10, 30000), self.length))


class TestNaturalNeighbor:
    @pytest.mark.unittest
    def test_natural_neighbor(self):
        # define some data to interpolate
        data = DataForTests(length=50)
        # initialize class
        interpolator = NaturalNeighbor()
        assert interpolator
        # interpolate
        plt.clf()
        plt.plot()
        plt.scatter(data.points_two_d.T[0], data.points_two_d.T[1], c=data.value)
        interpolator.interpolate(data.points_two_d, data.value)
        # create more points for test
        test_data = DataForTests(length=20)
        interpolator.predict(test_data.points_two_d)
        assert len(interpolator.zn) == 20

    @pytest.mark.unittest
    def test_natural_neighbor_3d(self):
        # define some data to interpolate
        data = DataForTests(length=50)
        # initialize class
        interpolator = NaturalNeighbor()
        assert interpolator
        # interpolate
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            data.points_three_d.T[0],
            data.points_three_d.T[1],
            data.points_three_d.T[2],
            c=data.value,
        )
        interpolator.interpolate(data.points_three_d, data.value)
        # create more points for test
        test_data = DataForTests(length=20)
        interpolator.predict(test_data.points_three_d)
        assert len(interpolator.zn) == 20


class TestCustomKriging:
    @pytest.mark.unittest
    def test_kriging_3d(self):
        # define some data to interpolate
        data = DataForTests(length=50)
        # initialize class
        interpolator = CustomKriging(two_d=False)
        interpolator.variogram_model = VariogramModel.gaussian
        interpolator.variogram_parameters.nugget = 40819929
        interpolator.variogram_parameters.range = 38
        interpolator.variogram_parameters.sill = 38020807
        assert interpolator
        # interpolate
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            data.points_three_d.T[0],
            data.points_three_d.T[1],
            data.points_three_d.T[2],
            c=data.value,
        )
        interpolator.interpolate(data.points_three_d, data.value)
        # create more points for test
        test_data = DataForTests(length=20)
        interpolator.predict(test_data.points_three_d)
        assert len(interpolator.zn) == 20

    @pytest.mark.unittest
    def test_kriging_2d(self):
        # define some data to interpolate
        data = DataForTests(length=50)
        # initialize class
        interpolator = CustomKriging(two_d=True)
        interpolator.variogram_model = VariogramModel.gaussian
        interpolator.variogram_parameters.nugget = 40819929
        interpolator.variogram_parameters.range = 38
        interpolator.variogram_parameters.sill = 38020807
        assert interpolator
        # interpolate
        plt.clf()
        plt.plot()
        plt.scatter(
            data.points_two_d.T[0],
            data.points_two_d.T[1],
            c=data.value,
        )
        interpolator.interpolate(data.points_two_d, data.value)
        # create more points for test
        test_data = DataForTests(length=20)
        interpolator.predict(test_data.points_two_d)
        assert len(interpolator.zn) == 20


class TestInverseDistance:
    @pytest.mark.unittest
    def test_inverse_distance(self):
        # define some data to interpolate
        data = DataForTests(length=50)
        # initialize class
        interpolator = InverseDistance()
        assert interpolator
        # interpolate
        plt.clf()
        plt.plot()
        plt.scatter(data.points_two_d.T[0], data.points_two_d.T[1], c=data.value)
        interpolator.interpolate(data.points_two_d, data.value)
        # create more points for test
        test_data = DataForTests(length=20)
        interpolator.predict(test_data.points_two_d)
        assert len(interpolator.zn) == 20
    @pytest.mark.unittest
    def test_inverse_distance_3d(self):
        # define some data to interpolate
        data = DataForTests(length=50)
        # initialize class
        interpolator = InverseDistance()
        assert interpolator
        # interpolate
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            data.points_three_d.T[0],
            data.points_three_d.T[1],
            data.points_three_d.T[2],
            c=data.value,
        )
        interpolator.interpolate(data.points_three_d, data.value)
        # create more points for test
        test_data = DataForTests(length=20)
        interpolator.predict(test_data.points_three_d)
        assert len(interpolator.zn) == 20