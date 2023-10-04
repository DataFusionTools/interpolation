from core.data_input import Data, Geometry

from interpolation.interpolation import *
from interpolation.interpolation_utils import InterpolationUtils
from interpolation.interpolation_inv_distance_per_depth import InverseDistancePerDepth

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import distance
from dataclasses import dataclass, field
from typing import List
import math
from scipy.stats import linregress
from sklearn import preprocessing


def get_values_for_inverse_dist_per_depth(dataclass_list: List[Data], value_name:str):
    """
    Function that extracts the values needed for the inverse distance per depth interpolation method

    :param dataclass_list: List of dataclasses that contain the data
    :param value_name: Name of the variable that should be interpolated

    :returns: coordinates, training_data, depth_data
    """
    # get coordinate list
    coordinates = [[datapoint.location.x, datapoint.location.y] for datapoint in dataclass_list]
    # get training data
    training_data = [datapoint.get_variable(value_name).value for datapoint in dataclass_list]
    # get depth data
    depth_data = [datapoint.independent_variable.value for datapoint in dataclass_list]
    return coordinates, training_data, depth_data


@dataclass
class Interpolate2DSlice:
    """
    Class that contains methods that facilitate the extraction of a 2d slice from a
    dataclass.

    :param interpolation_method_surface: Interpolation method selected from
       interpolating the top and bottom surfaces
    :param griddata_interpolation_method: Interpolation method to determine
        z coordinates of the top and bottom meshgrid select between the {linear, nearest, cubic}
        methods. linear is the default.
    """

    interpolation_method_surface: BaseClassInterpolation = field(
        default_factory=Nearest
    )
    griddata_interpolation_method: str = "linear"

    def get_2d_slice_per_depth_inverse_distance(
        self,
        location_1: Geometry,
        location_2: Geometry,
        data: List[Data],
        interpolate_variable: str,
        number_of_points: int,
        number_of_independent_variable_points: int,
        interpolation_method: InverseDistancePerDepth = None,
        top_surface: Union[List, None] = None,
        bottom_surface: Union[List, None] = None,
    ):
        """
        Fuction that produces an interpolated 2d slice of a 3d domain. This function uses the inverse distance per
        depth method to interpolate the 2d slice. In this case the interpolations method is of type InverseDistancePerDepth
        :class:`datafusiontools.interpolation.interpolation_inv_distance_per_depth.InverseDistancePerDepth`. This method
        uses the xy values of the list of dataclasses to find the n nearest points to the interpolation point. Then
        the n nearest data points are used to interpolate the variable of interest. The interpolation is done for each
        depth value. Returns the interpolated 2d slice and the variance of the interpolation.


        :param location_1: Starting point of the generated 2d slice
        :param location_2: Ending point of the generated 2d slice
        :param data: dataset with all available data
        :param interpolate_variable: Variable that the 2d slice will be interpolated for
        :param number_of_points: Number of point that should be available in the xy axis of the slice
        :param number_of_independent_variable_points: Number of points available in the z direction of the slice
        :param interpolation_method: Method that should be used for the interpolation of the variable
        :param top_surface: Optional input of the top surface of the slice
        :param bottom_surface: Optional input of the bottom surface of the slice

        :returns: points_2d_slice, results_2d_slice, variance
        """
        closest_bottom, closest_top = self.get_top_and_bottom_points(
            location_1, location_2, data, number_of_points, top_surface, bottom_surface
        )
        # get the data for training the interpolation model
        coordinates, training_data, depth_data = get_values_for_inverse_dist_per_depth(data, interpolate_variable)
        # loop through the closest points and interpolate
        points_2d_slice, results_2d_slice, variance = [], [], []
        for counter, top_point in enumerate(closest_top):
            bottom_point = closest_bottom[counter]
            if not (np.isnan(top_point[-1]) or np.isnan(bottom_point[-1])):
                # define independent variable discritisation
                z_prediction = np.linspace(
                    bottom_point[2], top_point[2], number_of_independent_variable_points
                )
                points = [[top_point[0], top_point[1], z] for z in z_prediction]
                # set interpolation method
                interpolation_method.interpolate(coordinates, training_data, depth_data, )
                # predict by giving a 2d location
                interpolation_method.predict_no_extrapolation(top_point[:2], z_prediction)
                # save results
                points_2d_slice.append(points)
                results_2d_slice.append(interpolation_method.zn)
                variance.append(interpolation_method.var)
        return points_2d_slice, results_2d_slice, variance

    def get_2d_slice_extra(
        self,
        location_1: Geometry,
        location_2: Geometry,
        data: List[Data],
        interpolate_variable: str,
        number_of_points: int,
        number_of_independent_variable_points: int,
        interpolation_method: BaseClassInterpolation = InverseDistance(
            nb_near_points=10
        ),
        top_surface: Union[List, None] = None,
        bottom_surface: Union[List, None] = None,
    ):
        """
        Function that produces an interpolated 2d slice of a 3d domain. In this case the user can choose the
        interpolation method. The interpolation method should be of type BaseClassInterpolation. This method
        uses the xyz values of the list of dataclasses to find the n nearest points to the interpolation point.


        :param location_1: Starting point of the generated 2d slice
        :param location_2: Ending point of the generated 2d slice
        :param data: dataset with all available data
        :param interpolate_variable: Variable that the 2d slice will be interpolated for
        :param number_of_points: Number of point that should be available in the xy axis of the slice
        :param number_of_independent_variable_points: Number of points available in the z direction of the slice
        :param interpolation_method: Method that should be used for the interpolation of the variable
        :param top_surface: Optional input of the top surface of the slice
        :param bottom_surface: Optional input of the bottom surface of the slice

        :returns: points_2d_slice, results_2d_slice
        """
        closest_bottom, closest_top = self.get_top_and_bottom_points(
            location_1, location_2, data, number_of_points, top_surface, bottom_surface
        )
        # check if interpolated value is string and if so encode
        #encoded = False
        #if self.get_type_of_variable(data, interpolate_variable) == str:
        #    encoded = True
        #    data = self.encode_string_data(data, interpolate_variable)

        # make interpolation model
        (
            value_interpolation,
            points_interpolation,
        ) = InterpolationUtils.get_data_for_interpolation(
            data,
            value_name=interpolate_variable,
            use_independent_variable_as_z=True,
        )
        # create model
        interpolation_method.interpolate(
            training_points=points_interpolation, training_data=value_interpolation
        )
        points_2d_slice, results_2d_slice = [], []
        for counter, top_point in enumerate(closest_top):
            bottom_point = closest_bottom[counter]
            if not (np.isnan(top_point[-1]) or np.isnan(bottom_point[-1])):
                # define independent variable discritisation
                z_prediction = np.linspace(
                    bottom_point[2], top_point[2], number_of_independent_variable_points
                )
                points = [[top_point[0], top_point[1], z] for z in z_prediction]
                points_2d_slice.append(points)
                interpolation_method.predict(points)
                results_2d_slice.append(interpolation_method.zn)
        if encoded:
            results_2d_slice = np.array(results_2d_slice).astype(int)
            results_2d_slice = [self.encoder.inverse_transform(result_row) for result_row in results_2d_slice]
        return points_2d_slice, results_2d_slice

    def encode_string_data(self,data: List[Data], interpolate_variable: str):
        """
        Function that encodes string data for interpolation.
        
        :param data: dataset with all available data
        :param interpolate_variable: Variable that the 2d slice will be interpolated for        

        :returns: Encoded data

        """
        self.encoder = preprocessing.LabelEncoder()
        # get all data
        flat_data = []
        for data_point in data:
            variable  = data_point.get_variable(interpolate_variable)
            flat_data += variable.value
        self.encoder.fit(np.append(flat_data, "unknown"))
        target_label = self.encoder.classes_  # classes
        for counter, data_point in enumerate(data):
            variable  = data_point.get_variable(interpolate_variable)
            target_encoded = self.encoder.transform(variable.value)
            data_point.update_variable(interpolate_variable, target_encoded)
            data[counter] = data_point
        return data

    def get_type_of_variable(self, data: List[Data], interpolate_variable: str):
        data_types = set()

        for data_point in data:
            variable = data_point.get_variable(interpolate_variable)
            if variable:
                data_types.add(type(variable.value))

        if len(data_types) == 1:
            return data_types.pop()
        else:
            raise ValueError(f"Data list provided has inconsistent types for variable {interpolate_variable}.")

    def get_user_defined_surface(
        self,
        location_1: Geometry,
        location_2: Geometry,
        number_of_points: int,
        surface: np.array,
    ):
        """
        Function that interpolates the user defined surface line to the number of points needed for the slice

        :param location_1: Starting point of the generated 2d slice
        :param location_2: Ending point of the generated 2d slice
        :param number_of_points: Number of point that should be available in the xy axis of the slice
        :param surface: user defined list of coordinates

        """
        if math.isclose(location_1.x, location_2.x):
            slope_slice_request = linregress(
                [location_1.y, location_2.y], [location_1.x, location_2.x]
            )
            slope_surface = linregress(surface.T[1], surface.T[0])
        else:
            slope_slice_request = linregress(
                [location_1.x, location_2.x], [location_1.y, location_2.y]
            )
            slope_surface = linregress(surface.T[0], surface.T[1])
        if not (math.isclose(slope_slice_request.slope, slope_surface.slope, abs_tol=1E-10)):
            raise ValueError(
                f"The surface selected does not have the same slope as the points selected for the slice"
            )
        if len(surface) == number_of_points:
            return surface
        else:
            # create interpolation to collect points
            interpolator = InverseDistance(nb_near_points=2)
            interpolator.interpolate(surface[:, :2], surface[:, -1])
            if math.isclose(location_1.x, location_2.x):
                y_prediction = np.linspace(location_1.y, location_2.y, number_of_points)
                if location_1.y < location_2.y:
                    x_prediction = np.interp(
                        y_prediction,
                        [location_1.y, location_2.y],
                        [location_1.x, location_2.x],
                    )
                else:
                    x_prediction = np.interp(
                        y_prediction,
                        [location_2.y, location_1.y],
                        [location_2.x, location_1.x],
                    )
            else:
                x_prediction = np.linspace(location_1.x, location_2.x, number_of_points)
                if location_1.x < location_2.x:
                    y_prediction = np.interp(
                        x_prediction,
                        [location_1.x, location_2.x],
                        [location_1.y, location_2.y],
                    )
                else:
                    y_prediction = np.interp(
                        x_prediction,
                        [location_2.x, location_1.x],
                        [location_2.y, location_1.y],
                    )
            prediction_points = np.array([x_prediction, y_prediction]).T
            interpolator.predict(prediction_points)
            return np.array([x_prediction, y_prediction, interpolator.zn]).T

    def get_top_and_bottom_points(
        self,
        location_1: Geometry,
        location_2: Geometry,
        data: List[Data],
        number_of_points: int,
        top_surface: Union[List, None, np.array] = None,
        bottom_surface: Union[List, None, np.array] = None,
    ):
        """
        Function that gets the top and bottoms points of the slice depending on the input of the user

        :param location_1: Starting point of the generated 2d slice
        :param location_2: Ending point of the generated 2d slice
        :param data: dataset with all available data
        :param interpolate_variable: Variable that the 2d slice will be interpolated for
        :param number_of_points: Number of point that should be available in the xy axis of the slice
        :param number_of_independent_variable_points: Number of points available in the z direction of the slice
        :param interpolation_method: Method that should be used for the interpolation of the variable
        :param top_surface: Optional input of the top surface of the slice
        :param bottom_surface: Optional input of the bottom surface of the slice
        """
        if top_surface is not None and bottom_surface is not None:
            closest_top = self.get_user_defined_surface(
                location_1, location_2, number_of_points, top_surface
            )
            closest_bottom = self.get_user_defined_surface(
                location_1, location_2, number_of_points, bottom_surface
            )
        else:
            closest_bottom, closest_top = self.calculate_top_and_bottom_points(
                location_1, location_2, data, number_of_points
            )
            if top_surface is not None:
                closest_top = self.get_user_defined_surface(
                    location_1, location_2, number_of_points, top_surface
                )
            if bottom_surface is not None:
                closest_bottom = self.get_user_defined_surface(
                    location_1, location_2, number_of_points, bottom_surface
                )
        return closest_bottom, closest_top

    def calculate_top_and_bottom_points(
        self,
        location_1: Geometry,
        location_2: Geometry,
        data: List[Data],
        number_of_points: int,
    ):
        """Function that return the top and bottom coordinates of the 2d slice.

        :param location_1: Starting point of the generated 2d slice
        :param location_2: Ending point of the generated 2d slice
        :param data: dataset with all available data
        :param number_of_points: Number of point that should be available in the xy axis of the slice
        """
        # get bottom and top locations
        top = np.array(
            [
                [
                    datapoint.location.x,
                    datapoint.location.y,
                    datapoint.independent_variable.value[0],
                ]
                for datapoint in data
            ]
        )
        bot = np.array(
            [
                [
                    datapoint.location.x,
                    datapoint.location.y,
                    datapoint.independent_variable.value[-1],
                ]
                for datapoint in data
            ]
        )
        # run grid analysis
        extrapolation_spots = self.get_plane(
            location_1.x, location_2.x, location_1.y, location_2.y, number_of_points
        )
        grid_top, grid_bottom = self.interpolation_analysis(
            extrapolation_spots, top, bot, number_of_points
        )
        # get intersection of line with the mesh
        coordinates_mesh_top = np.vstack(
            [grid_top[0].ravel(), grid_top[1].ravel(), grid_top[2].ravel()]
        ).T
        coordinates_mesh_bottom = np.vstack(
            [grid_bottom[0].ravel(), grid_bottom[1].ravel(), grid_bottom[2].ravel()]
        ).T
        # get the closest points
        closest_top, closest_bottom = [], []
        # create x,y line
        if math.isclose(location_1.x, location_2.x):
            y_prediction = np.linspace(location_1.y, location_2.y, number_of_points)
            min_y = min([location_1.y, location_2.y])
            max_y = max([location_1.y, location_2.y])                
            x_prediction = np.interp(
                y_prediction,
                [min_y, max_y],
                [location_1.x, location_2.x],
            )
        else:
            x_prediction = np.linspace(location_1.x, location_2.x, number_of_points)
            min_x = min([location_1.x, location_2.x])
            max_x = max([location_1.x, location_2.x])                
            y_prediction = np.interp(
                x_prediction,
                [min_x, max_x],
                [location_1.y, location_2.y],
            )
        slice_line = np.array((list(x_prediction), list(y_prediction))).T
        for point in slice_line:
            closest_index_top = self.closest_node_index(
                point, coordinates_mesh_top[:, :2]
            )
            closest_top.append(coordinates_mesh_top[closest_index_top])
            closest_index_bottom = self.closest_node_index(
                point, coordinates_mesh_bottom[:, :2]
            )
            closest_bottom.append(coordinates_mesh_bottom[closest_index_bottom])
        return closest_bottom, closest_top

    def interpolation(self, data, number_of_points):
        """Function that creates a 3d mesh grid.

        :param data: dataset with all available data
        :param number_of_points: Number of point that should be available in the xy axis of the slice
        """
        min_x = min(data[:, 0])
        max_x = max(data[:, 0])
        min_y = min(data[:, 1])
        max_y = max(data[:, 1])
        diagonal = int(math.dist((min_x, min_y), (max_x, max_y)))
        gridx, gridy = np.mgrid[
            min_x : max_x : (diagonal + number_of_points) * 1j, min_y : max_y : (diagonal + number_of_points) * 1j
        ]
        gridz = griddata(
            data[:, :2],
            data[:, 2],
            (gridx, gridy),
            method=self.griddata_interpolation_method,
        )
        return gridx, gridy, gridz

    def get_plane(self, xl, xu, yl, yu, i):
        """Function that returns a collection of points representing a plane used as a
        search domain.

        :param xl: x coordinate 1rst point
        :param xu: x coordinate 2nd point
        :param yl: y coordinate 1rst point
        :param yu: y coordinate 2nd point
        :param i: discritization of points produced
        """
        if xl == xu:
            xx = [xu]
        else:
            xx = np.arange(xl, xu, (xu - xl) / i)
        if yl == yu:
            yy = [yu]
        else:
            yy = np.arange(yl, yu, (yu - yl) / i)
        extrapolation_spots = np.zeros((len(xx) * len(yy), 2))
        count = 0
        for i in xx:
            for j in yy:
                extrapolation_spots[count, 0] = i
                extrapolation_spots[count, 1] = j
                count += 1
        return extrapolation_spots

    def interpolation_analysis(self, extrapolation_spots, top, bot, number_of_points):
        """Function that performs the interpolation analysis to create a mesh grid of
        the top and bottom surfaces.

        :param extrapolation_spots: extrapolated 3d plane space points
        :param top: known top points of the 3d space
        :param bottom: known bottom points of the 3d space
        :param number_of_points: Number of point that should be available in the xy axis of the slice
        """
        top_extra = self.extrapolation(top, extrapolation_spots)
        bot_extra = self.extrapolation(bot, extrapolation_spots)
        gridx_top, gridy_top, gridz_top = self.interpolation(
            top_extra, number_of_points
        )
        gridx_bot, gridy_bot, gridz_bot = self.interpolation(
            bot_extra, number_of_points
        )
        return (gridx_top, gridy_top, gridz_top), (gridx_bot, gridy_bot, gridz_bot)

    def extrapolation(self, data, extrapolation_spots):
        """Function that extrapolates points from 2d to 3d.

        :param data: already existing 3d points
        :param extrapolation_spots: 2d points that the z irectio should be extrapolated
        """
        new_points = np.zeros((len(extrapolation_spots), 3))
        new_points[:, 0] = extrapolation_spots[:, 0]
        new_points[:, 1] = extrapolation_spots[:, 1]

        self.interpolation_method_surface.interpolate(
            training_points=data[:, :2], training_data=data[:, -1]
        )
        self.interpolation_method_surface.predict(new_points[:, :2])
        for i in range(len(extrapolation_spots)):
            new_points[i, 2] = self.interpolation_method_surface.zn[i]
        combined = np.concatenate((data, new_points))
        return combined

    def closest_node_index(self, node, nodes):
        """Function that returns an index that corresponds to the closest point in list.

        :param node: reference point
        :param nodes: list of points that are the search array
        """
        closest_index = distance.cdist([node], nodes).argmin()
        return closest_index
