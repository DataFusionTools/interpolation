import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy import interpolate
import pykrige
from dataclasses import dataclass, asdict, field
from typing import List, Union, Dict, Optional
from abc import abstractmethod
import numpy as np
from enum import Enum
from core.base_class import BaseClass


class VariogramModel(Enum):
    linear = "linear"
    power = "power"
    gaussian = "gaussian"
    spherical = "spherical"
    exponential = "exponential"
    hole_effect = "hole-effect"


@dataclass
class VariogramParameters:

    slope: Optional[float] = None
    nugget: Optional[float] = None
    scale: Optional[float] = None
    exponent: Optional[float] = None
    nugget: Optional[float] = None
    sill: Optional[float] = None
    psill: Optional[float] = None
    range: Optional[float] = None


@dataclass
class BaseClassInterpolation(BaseClass):
    tree: Union[List, None, np.array] = None
    zn: Union[List, None, np.array] = None
    training_data: Union[List, None, np.array] = None
    training_points: Union[List, None, np.array] = None

    @abstractmethod
    def interpolate(self):
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )

    @abstractmethod
    def predict(self):
        raise NotImplementedError(
            "The method should be implemented in concrete classes."
        )


@dataclass
class Nearest(BaseClassInterpolation):
    def interpolate(self, training_points: np.array, training_data: np.array):
        """
        Define the KDtree

        This interpolation is done with `SciPy interpolate.NearestNDInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html>`_.

        :param training_points: array with the training points
        :param training_data: data at the training points
        :return:
        """
        # assign to variables
        self.training_points = training_points  # training points
        self.training_data = training_data  # data at the training points
        # define interpolation function
        self.interpolating_function = interpolate.NearestNDInterpolator(
            self.training_points, self.training_data
        )
        # create KDtree
        self.tree = cKDTree(training_points)
        return

    def predict(self, prediction_points):
        """
        Perform interpolation with nearest neighbors method

        :param prediction_points: prediction points
        :return:
        """
        # compute closest distance and index of the closest index
        dist, idx = self.tree.query(prediction_points)
        self.zn = []
        # create interpolation for every point
        for i in range(len(prediction_points)):
            # interpolate
            self.zn.append(self.training_data[idx[i]])

        self.zn = np.array(self.zn)

        return


@dataclass
class InverseDistance(BaseClassInterpolation):
    """
    Inverse distance interpolator
    """

    nb_near_points: int = 6
    power: float = 1.0
    tol: float = 1e-9
    var: Union[List, None, np.array] = None

    def interpolate(self, training_points, training_data):
        """
        Define the KDtree

        :param training_points: array with the training points
        :param training_data: data at the training points
        :return:
        """

        # assign to variables
        self.training_points = np.array(training_points)  # training points
        self.training_data = np.array(training_data)  # data at the training points

        # compute Euclidean distance from grid to training
        self.tree = cKDTree(self.training_points)

        return

    def predict(self, prediction_points):
        """
        Perform interpolation with inverse distance method

        :param prediction_points: prediction points
        :return:
        """

        # get distances and indexes of the closest nb_points
        dist, idx = self.tree.query(prediction_points, self.nb_near_points)
        dist += self.tol  # to overcome division by zero
        self.zn = []

        # create interpolation for every point
        for i in range(len(prediction_points)):
            # compute weights
            data = self.training_data[idx[i]]

            # interpolate
            self.zn.append(
                np.sum(data.T / dist[i] ** self.power)
                / np.sum(1.0 / dist[i] ** self.power)
            )

        self.zn = np.array(self.zn)

        return


@dataclass
class NaturalNeighbor(BaseClassInterpolation):

    interp: Union[List, None, np.array] = None

    def interpolate(self, training_points, training_data):
        """
        Define the interpolator

        This interpolation is done with `SciPy interpolate.NearestNDInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html>`_.

        :param training_points: array with the training points
        :param training_data: data at the training points
        :return:
        """

        # assign to variables
        self.training_points = training_points  # training points
        self.training_data = training_data  # data at the training points

        return

    def predict(self, prediction_points):
        """
        Perform interpolation with natural neighbors method

        :param prediction_points: prediction points
        :return:
        """

        zn = []
        prediction_points = np.array(prediction_points)
        for i in range(len(prediction_points)):
            new_points = np.vstack([self.training_points, prediction_points[i].T])
            tri = Delaunay(new_points)
            # Find index of prediction point
            pindex = np.where(np.all(tri.points == prediction_points[i].T, axis=1))[0][
                0
            ]
            # find neighbours
            neig_idx = tri.vertex_neighbor_vertices[1][
                tri.vertex_neighbor_vertices[0][pindex] : tri.vertex_neighbor_vertices[
                    0
                ][pindex + 1]
            ]

            # get the coordinates of the neighbours
            coords_neig = [tri.points[j] for j in neig_idx]
            # compute Euclidean distance
            dist = [np.linalg.norm(prediction_points[i] - j) for j in coords_neig]
            # find data of the neighbours
            idx_coords_neig = []
            for j in coords_neig:
                idx_coords_neig.append(
                    np.where(np.all(self.training_points == j, axis=1))[0][0]
                )

            # get data of the neighbours
            data_neig = [np.array(self.training_data[j]) for j in idx_coords_neig]
            # compute weights
            zn_aux = []
            for ii in range(len(data_neig)):
                aux = data_neig[ii] * dist[ii] / np.sum(dist)
                zn_aux.append(aux)
            zn.append(np.sum(np.array(zn_aux), axis=0))

        self.zn = zn

        return


@dataclass
class CustomKriging(BaseClassInterpolation):
    two_d: bool = True
    variogram_model: VariogramModel = VariogramModel.linear
    variogram_parameters: VariogramParameters = field(default_factory=VariogramParameters)

    def interpolate(self, training_points, training_data):
        # assign to variables
        if self.two_d:
            self.training_points = training_points  # training points
            self.training_data = training_data  # data at the training points
            self.interpolating_function = pykrige.ok.OrdinaryKriging(
                self.training_points.T[0],
                self.training_points.T[1],
                training_data,
                variogram_model=self.variogram_model.value,
                variogram_parameters={
                    k: v
                    for k, v in asdict(self.variogram_parameters).items()
                    if v is not None
                },
                verbose=False,
                enable_plotting=False,
                nlags=20,
            )
        else:
            self.training_points = training_points  # training points
            self.training_data = training_data  # data at the training points
            self.interpolating_function = pykrige.ok3d.OrdinaryKriging3D(
                self.training_points.T[0],
                self.training_points.T[1],
                self.training_points.T[2],
                training_data,
                variogram_model=self.variogram_model.value,
                variogram_parameters={
                    k: v
                    for k, v in asdict(self.variogram_parameters).items()
                    if v is not None
                },
                verbose=False,
                enable_plotting=False,
                nlags=20,
            )

    def predict(self, prediction_points):
        """
        Perform interpolation with Kriging method

        :param prediction_points: prediction points
        :return:
        """
        if self.two_d:
            self.zn, ss = self.interpolating_function.execute(
                "points", prediction_points.T[0], prediction_points.T[1]
            )
        else:
            self.zn, ss = self.interpolating_function.execute(
                "points",
                prediction_points.T[0],
                prediction_points.T[1],
                prediction_points.T[2],
            )
