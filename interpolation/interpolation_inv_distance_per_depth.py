from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from typing import List, Union


@dataclass
class InverseDistancePerDepth:
    """
    Inverse distance interpolator per depth or some other set independent variable. The interpolation is performed
    using the inverse distance method. The mean and variance are computed based

    :param nb_near_points: (optional) number of k-nearest neighbours used for interpolation. Default is 6
    :param power: (optional) power of the inverse. Default is 1
    :param tol: (optional) tolerance added to the point distance to overcome division by zero. Default is 1e-9
    :param default_cov: (optional) default covariance for the case of only one datapoint. Default is 10

    Extra attributes:

    - tree: KDtree with nearest neighbors
    - zn: interpolation results
    - var: interpolation variance
    - training_data: training data
    - training_points: training points
    - depth_data: training points depth
    - depth_prediction: interpolation depth

    """
    nb_near_points: int = 6
    power: int = 1
    tol: float = 1e-9
    default_cov: float = 10.
    tree: cKDTree = None
    zn: Union[List, None, np.array] = None
    var: Union[List, None, np.array] = None
    training_data: Union[List, None, np.array] = None
    training_points: Union[List, None, np.array] = None
    depth_data: Union[List, None, np.array] = None
    depth_prediction: Union[List, None, np.array] = None

    def interpolate(self,
                    training_points: Union[List, np.array],
                    training_data: Union[List, np.array],
                    depth_points: Union[List, np.array],):
        """
        Function that defines the training data and the interpolation points for the inverse distance method.
        The training data are the data at the training points and the interpolation points are the points where
        the interpolation is performed.

        :param training_points: array with the training points
        :param training_data: data at the training points
        :param depth_points: depth at the training points
        """

        # assign to variables
        self.training_points = np.array(training_points)  # training points
        self.training_data = training_data  # data at the training points
        self.depth_data = depth_points # depth from the training points

        # compute Euclidean distance from grid to training
        self.tree = cKDTree(self.training_points)

    @staticmethod
    def merge_values_within_bound(values, bound):
        merged_values = []
        for value in values:
            if not merged_values or abs(value - merged_values[-1]) > bound:
                merged_values.append(value)
        return merged_values

    def predict_no_extrapolation(self,
                                 prediction_point: Union[List, np.array],
                                 depth_prediction: Union[List, np.array],
                                 point: bool = False):

        zn = []
        var = []
        # find tops of the training data so that we can identify the extrapolating boundaries
        tops = sorted([depth[0] for depth in self.depth_data], reverse=True)
        bottoms = sorted([depth[-1] for depth in self.depth_data], reverse=True)
        # merge the tops and bottoms if the values are within 1 meter of each other
        tops = self.merge_values_within_bound(tops, 1)
        bottoms = self.merge_values_within_bound(bottoms, 1)
        # create interpolation boundaries for the case of extrapolating
        boundaries = tops + bottoms
        # remove duplicates
        boundaries = sorted(list(dict.fromkeys(boundaries)), reverse=True)
        # loop pairwise through the boundaries
        local_trees = []
        for top_boundary, bottom_boundary in zip(boundaries, boundaries[1:]):
            # collect prediction depths that is within the boundaries
            # the last boundary should include the bottom of depth prediction
            if bottom_boundary == boundaries[-1]:
                local_depth_prediction = [depth for depth in depth_prediction if
                                          top_boundary >= depth >= bottom_boundary]
            else:
                local_depth_prediction = [depth for depth in depth_prediction if
                                          top_boundary >= depth > bottom_boundary]
            if bool(local_depth_prediction):
                # collect training points that their depth is within the boundaries
                local_training_points = []
                local_training_data = []
                local_depth_data = []
                for index, point_train in enumerate(self.training_points):
                    training_depth = self.depth_data[index]
                    is_within_boundary = np.any([top_boundary > depth > bottom_boundary for depth in training_depth])
                    # truncate the training points and data
                    if is_within_boundary:
                        local_training_points.append(point_train)
                        # collect indexes
                        index_train_collection = [train_index for train_index, depth in enumerate(training_depth) if top_boundary >= depth >= bottom_boundary]
                        # truncate the training data
                        local_training_data.append(np.array(self.training_data[index])[index_train_collection])
                        # truncate the training depth
                        local_depth_data.append(np.array(training_depth)[index_train_collection])
                # create new KDTree
                local_trees.append(cKDTree(local_training_points))
                # predict
                self.predict_with_custom_tree(local_trees[-1], prediction_point, local_training_data, local_depth_data, local_depth_prediction, point)

                zn.append(self.zn)
                var.append(self.var)
        # flatten the lists
        zn = [item for sublist in zn for item in sublist]
        var = [item for sublist in var for item in sublist]
        # are there points that are outside the boundaries they should be extrapolated
        top_extras = [depth for depth in depth_prediction if depth > boundaries[0]]
        zn_top = [zn[0]] * len(top_extras)
        var_top = [var[0]] * len(top_extras)
        bottom_extras = [depth for depth in depth_prediction if depth < boundaries[-1]]
        zn_bottom = [zn[-1]] * len(bottom_extras)
        var_bottom = [var[-1]] * len(bottom_extras)
        # append the extrapolated points
        zn = zn_top + zn + zn_bottom
        var = var_top + var + var_bottom
        # return the results
        self.zn = zn
        self.var = var

    def predict(self, prediction_point: Union[List, np.array], depth_prediction: Union[List, np.array], point: bool = False):
        """
        Perform interpolation with inverse distance method .The mean and variance are computed based
        on :cite:p:`deutsch_2009`, :cite:p:`calle_1`, :cite:p:`calle_2`.

        :param prediction_point: prediction points
        :param depth_prediction: depth at the prediction points
        :param point: if True, the interpolation is performed at a single point
        """
        self.predict_with_custom_tree(self.tree,
                                      prediction_point,
                                      self.training_data,
                                      self.depth_data,
                                      depth_prediction,
                                      point)


    def predict_with_custom_tree(self,
                tree: cKDTree,
                prediction_point: Union[List, np.array],
                training_data: Union[List, np.array],
                depth_data: Union[List, np.array],
                depth_prediction: Union[List, np.array],
                point: bool = False):
        """
        Perform interpolation with inverse distance method .The mean and variance are computed based
        on :cite:p:`deutsch_2009`, :cite:p:`calle_1`, :cite:p:`calle_2`.

        :param prediction_point: prediction points
        :param training_data: data at the training points
        :param depth_data: depth from the training points
        :param depth_prediction: depth at the prediction points
        :param point: (optional) boolean for the case of being a single point
        """
        if len(training_data) < self.nb_near_points:
            nb_near_points = len(prediction_point)
        else:
            nb_near_points = self.nb_near_points
        # get distances and indexes of the closest nb_points
        dist, idx = tree.query(prediction_point, nb_near_points)
        dist += self.tol  # to overcome division by zero
        dist = np.array(dist).reshape(nb_near_points)
        idx = np.array(idx).reshape(nb_near_points)

        # for every dataset
        point_aver = []
        point_val = []
        point_var = []
        point_depth = []
        for p in range(nb_near_points):
            # compute the weights
            wei = (1. / dist[p]**self.power) / np.sum(1. / dist ** self.power)
            # if single point
            if point:
                point_aver.append(training_data[idx[p]] * wei)
                point_val.append(training_data[idx[p]])
            # for multiple points
            else:
                try:
                    point_aver.append(np.log(training_data[idx[p]]) * wei)
                except:
                    print("error")
                point_val.append(np.log(training_data[idx[p]]))
            point_depth.append(depth_data[idx[p]])

        # compute average
        if point:
            zn = [np.sum(point_aver)]
        else:
            new = []
            for i in range(nb_near_points):
                f = interp1d(point_depth[i], point_aver[i], fill_value=(point_aver[i][-1], point_aver[i][0]), bounds_error=False)
                new.append(f(depth_prediction))
            zn = np.sum(np.array(new), axis=0)

        # compute variance
        if point:
            for p in range(nb_near_points):
                # compute the weighs
                wei = (1. / dist[p] ** self.power) / np.sum(1. / dist ** self.power)
                point_var.append((point_val[p] - zn) ** 2 * wei)
        else:

            # compute mean
            new = []
            # 1. for each nearest point p
            for i in range(nb_near_points):
                # 2.  The method first interpolates the training data values onto the prediction depths
                # using linear interpolation.
                f = interp1d(point_depth[i], point_val[i], fill_value=(point_val[i][-1], point_val[i][0]), bounds_error=False)
                new.append(f(depth_prediction))
            # compute variance
            for p in range(nb_near_points):
                # compute the weights
                wei = (1. / dist[p] ** self.power) / np.sum(1. / dist ** self.power)
                # compute var
                # 3.  It then computes the squared difference between the interpolated value new[p] and
                # the predicted value zn, and multiplies this squared difference by the weight wei for that
                # training point. This gives the variance contribution for that training point.
                point_var.append((new[p] - zn) ** 2 * wei)
        # 4. The variance of the prediction is the sum of the variance contributions
        # for all the nearest neighbor training points.
        var = np.sum(np.array(point_var), axis=0)

        # add to variables
        if point:
            # update to normal parameters
            self.zn = zn
            self.var = var
        else:
            # update to lognormal parameters
            self.zn = np.exp(zn + var / 2)
            self.var = np.exp(2 * zn + var) * (np.exp(var) - 1)

        # if only 1 data point is available (var = 0 for all points) -> var is default value
        if nb_near_points == 1:
            self.var = np.full(len(self.var), (self.default_cov * np.array(self.zn)) ** 2)


