from typing import List, Union
import numpy as np

from core.data_input import Data



class InterpolationUtils:
    @staticmethod
    def get_points_from_one_dataclass(
        data_point: Data, use_independent_variable_as_z: bool
    ):

        if use_independent_variable_as_z:
            independent_variables = data_point.independent_variable.value
            local_point_list = []
            for independent_variable in independent_variables:
                local_point_list.append(
                    [
                        data_point.location.x,
                        data_point.location.y,
                        independent_variable,
                    ]
                )
            points_interpolation = local_point_list
        else:
            points_interpolation = [
                [data_point.location.x, data_point.location.y]
            ] * len(data_point.variables[0].value)
        return points_interpolation

    @staticmethod
    def get_data_for_interpolation(
        data: Union[Data, List[Data]],
        value_name: str,
        use_independent_variable_as_z: bool,
    ):
        """
        Static function that transforms inputted data to be used for the interpolation process
        """
        if isinstance(data, list):
            value_interpolation = []
            points_interpolation = []
            for data_point in data:
                points_of_data = InterpolationUtils.get_points_from_one_dataclass(
                    data_point, use_independent_variable_as_z
                )
                value_interpolation.extend(data_point.get_variable(value_name).value)
                points_interpolation.extend(points_of_data)
            value_interpolation = np.array(value_interpolation).flatten()
            points_interpolation = np.array(points_interpolation)
            return value_interpolation, points_interpolation
        else:
            points_interpolation = InterpolationUtils.get_points_from_one_dataclass(
                data, use_independent_variable_as_z
            )
            return np.array(data.get_variable(value_name).value), np.array(
                points_interpolation
            )

    @staticmethod
    def get_points_for_interpolation(
        data: Union[Data, List[Data]],
        use_independent_variable_as_z: bool,
    ):
        """
        Static function that transforms inputted data to points that can be used for the interpolation prediction
        """
        if isinstance(data, list):
            points_interpolation = []
            for data_point in data:
                points_of_data = InterpolationUtils.get_points_from_one_dataclass(
                    data_point, use_independent_variable_as_z
                )
                points_interpolation.append(points_of_data)
            points_interpolation = np.array(points_interpolation)
            points_interpolation = np.reshape(
                points_interpolation,
                (
                    points_interpolation.shape[0] * points_interpolation.shape[1],
                    points_interpolation.shape[2],
                ),
            )
            return points_interpolation
        else:
            points_interpolation = InterpolationUtils.get_points_from_one_dataclass(
                data, use_independent_variable_as_z
            )
            return np.array(points_interpolation)
