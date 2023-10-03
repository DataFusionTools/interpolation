import pytest
import numpy as np
import matplotlib.pyplot as plt

from interpolation.interpolation_inv_distance_per_depth import InverseDistancePerDepth


def getEquidistantPoints(p1, p2, parts):
    return list(zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1)))


class TestInverseDistancePerDepth:

    def test_mean_and_variance_output_1_point(self):

        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.array(4)
        # interpolate the top and bottom depth at this point
        interp = InverseDistancePerDepth(nb_near_points=len(data), power=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data, position)
        # predict
        interp.predict(testing.reshape(1, 1), position, point=True)

        # testing - mean
        assert np.allclose(interp.zn[0], data[4])
        # testing - var
        dist = position - testing + 1e-9
        var = []
        for i in range(len(position)):
            weight = (1. / dist[i] ** 1) / np.sum(1. / dist ** 1)
            var.append((data[i] - data[4]) ** 2 * weight)
        assert np.allclose(interp.var[0], np.sum(var))

    @pytest.mark.parametrize("pw", [1, 2, 3, 4, 5])
    def test_mean_and_variance_output_multiple_points(self, pw):
        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.random.rand(10) * 10
        # interpolate the top and bottom depth at this point
        interp = InverseDistancePerDepth(nb_near_points=len(data), power=pw)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data, position)

        # for each testing point
        for i, t in enumerate(testing):
            # predict
            interp.predict(t.reshape(1, 1), position, point=True)
            # result
            dist = np.abs((position - t) + 1e-9)
            val = np.sum(data / (dist ** pw)) / np.sum(1 / (dist ** pw))

            # testing - mean
            assert np.allclose(interp.zn[0], val)
            # testing - var
            var = []
            for k in range(len(position)):
                weight = (1. / dist[k] ** pw) / np.sum(1. / dist ** pw)
                var.append((data[k] - val) ** 2 * weight)
            assert np.allclose(interp.var[0], np.sum(var))

    @pytest.mark.parametrize("testing", [
                             (np.array([0.5, 0.5])),
                             (np.array([0, 0])),
                             (np.array([0.9, 0.9])),
                             (np.array([1.5, 1.5])),])
    def test_check_different_points(self, testing):

        position = np.array([[0, 0], [1, 1]])
        data = [np.ones(10), np.ones(10) * 2]
        # interpolate the top and bottom depth at this point
        interp = InverseDistancePerDepth(nb_near_points=len(data), power=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 2)), data,
                           [np.linspace(0, 10, 10), np.linspace(0, 10, 10)])
        # predict
        interp.predict(testing.reshape(1, 2), np.linspace(0, 10, 10))

        # compute correct lognormal parameters
        mean, var = self.compute_lognormal(position, testing, data)

        # testing - mean
        assert np.allclose(interp.zn, mean)
        # testing - var
        assert np.allclose(interp.var, var)

    @staticmethod
    def compute_lognormal(position, testing, data):

        # compute mean and var
        dist = []
        for i in position:
            dist.append(np.linalg.norm(position[i] - testing) + 1e-9)
        dist = np.array(dist)
        mean = []
        var = []
        for k in range(len(position)):
            weight = (1. / dist[k] ** 1) / np.sum(1. / dist ** 1)
            mean.append(np.log(data[k]) * weight)

        aux_m = np.sum(np.array(mean), axis=0)

        for k in range(len(position)):
            weight = (1. / dist[k] ** 1) / np.sum(1. / dist ** 1)
            var.append((np.log(data[k]) - aux_m) ** 2 * weight)

        aux_v = np.sum(np.array(var), axis=0)
        mean = np.exp(aux_m + aux_v / 2)
        var = np.exp(2 * aux_m + aux_v) * (np.exp(aux_v) - 1)

        return mean, var


    @pytest.mark.unittest
    def test_extrapolation(self):
        # create synthetic cpts data
        position = np.array([[0, 0.5], [0, 2.5], [0, 3.5], [0,5.5]])
        available_points = 50
        data = [np.ones(available_points) * 50, np.ones(available_points) * 2, np.ones(available_points) * 3, np.ones(available_points) * 50]
        depths = [np.linspace(-30, 0, available_points), np.linspace(-15, 4, available_points), np.linspace(-15, 4, available_points), np.linspace(-30, 0, available_points)]
        # flip depth list
        depths = [np.flip(d) for d in depths]
        # pick point in the middle of the geometry
        number_of_points = 50
        point_xy = np.array([0, 3])
        depth_prediction = np.linspace(-30, 4, number_of_points)
        # test interpolation
        interp = InverseDistancePerDepth(nb_near_points=3, power=1)
        interp.interpolate(position, data, depths)
        # predict
        interp.predict_no_extrapolation(point_xy.reshape(1, 2), depth_prediction)
        # check if the prediction is correct
        assert np.allclose(interp.zn[:5], 2)
        assert np.allclose(interp.zn[28:], 50)

    @pytest.mark.unittest
    def test_merge_values_within_bound(self):
        # create test list
        tops = [6.47, 6.46, 5.84, 1.95, 1.94, 1.87, 1.83, 1.8, 0.31, -0.51]
        bottoms = [-12.82898753894081, -12.865819032761312, -13.00213395638629, -14.435250783699056, -15.055284768211925, -15.608244604316544, -17.907825159914708, -18.299999999999997, -18.330000000000002, -18.95]
        # check tops
        merged_values = InverseDistancePerDepth.merge_values_within_bound(tops, 1)
        assert np.allclose(merged_values, [6.47, 1.95,  0.31])
        # check bottoms
        merged_values = InverseDistancePerDepth.merge_values_within_bound(bottoms, 1)
        assert np.allclose(merged_values, [-12.83, -14.43, -15.608,  -17.91, -18.95], atol=1e-2)
        print(merged_values)


        ## create points in xyz to interpolate
        #dike_xz_positions = np.array([[0,0], [1,0], [2,4], [4,4], [5,0], [6,0]])
        ## create point cloud until the depth of 30 meters
        #line_points = []
        ## create a grid of points up to a depth of 30 meters
        #for pair in zip(dike_xz_positions, dike_xz_positions[1:] + dike_xz_positions[:1]):
        #    line_points.append(getEquidistantPoints(pair[0], pair[1], 2))
        #line_points = np.reshape(np.array(line_points), (5 * 3, 2))
        #interpolation_points = [[], [], [], []]
        ## interpolate the top and bottom depth at this point
        #interp = InverseDistancePerDepth(nb_near_points=3, power=1)
        #
        #interp.interpolate(position, data, depths)
        #for point in line_points:
        #    sub_y = [point[0]] * 50
        #    sub_z = list(np.flip(np.linspace(-30, point[1], 50)))
        #    interpolation_points[0] += list(np.zeros_like(sub_z))
        #    interpolation_points[1] += sub_y
        #    interpolation_points[2] += sub_z
        #    # predict
        #    interp.predict_no_extrapolation([0, point[0]], sub_z)
        #    interpolation_points[-1] += list(interp.zn)
        ## plotting results
        #fig = plt.figure()
        #ax = plt.axes(projection='3d')
        ## add the cpts in the plot with their color
        #xs, ys, zs, cs = [], [], [], []
        #for i, p in enumerate(position):
        #    xs += [p[0]] * len(depths[i])
        #    ys += [p[1]] * len(depths[i])
        #    zs += list(depths[i])
        #    cs += list(data[i])
        #p1 = ax.scatter(xs, ys, zs, c=cs)
        #p = ax.scatter(interpolation_points[0], interpolation_points[1], interpolation_points[2], c=interpolation_points[3])
        #fig.colorbar(p1, ax=ax)
        #plt.show()
