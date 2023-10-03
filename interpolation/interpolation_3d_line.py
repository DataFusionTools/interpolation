
from core.data_input import Data
from interpolation.interpolation import *
from interpolation.interpolation_utils import InterpolationUtils
from spatial_utils.ahn_utils import SpatialUtils

@dataclass
class Interpolate3DLine:

    def get_3d_line(self,
        line: List,         
        data: List[Data],
        interpolate_variable: str,
        number_of_independent_variable_points: int,
        bottom_value : float,
        interpolation_method: BaseClassInterpolation = InverseDistance(
            nb_near_points=10
        )
        
        ):

        # define interpolator based on data given
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
        # get total AHN line
        spatial_utils = SpatialUtils()
        spatial_utils.get_ahn_surface_line(np.array(line))
        # loop through points to create 3d point field
        x , y, z = [], [], []
        for counter, location in enumerate(spatial_utils.AHN_data):
            z_field = list(np.linspace(location[2], bottom_value, number_of_independent_variable_points))
            z += z_field
            x += [location[0]]* len(z_field) 
            y += [location[1]]* len(z_field)
        interpolation_field = np.array([x,y, z]).T
        interpolation_method.predict(interpolation_field)
        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d') 
        # p3d = ax.scatter(interpolation_field.T[0], interpolation_field.T[1], interpolation_field.T[2], s=30, c=interpolation_method.# zn, marker='o')
        return interpolation_field, interpolation_method.zn



