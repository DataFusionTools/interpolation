import pytest
import numpy as np
import pickle
from sklearn import preprocessing
import math

from core.data_input import Data, Variable
from interpolation.interpolation_2d_slice import Interpolate2DSlice
from interpolation.interpolation import Nearest, InverseDistance
from core.data_input import Geometry
from spatial_utils.ahn_utils import SpatialUtils
from interpolation.interpolation_inv_distance_per_depth import InverseDistancePerDepth
from utils import TestUtils


class TestInterpolate2DSlice:
    def generate_fake_cpt(self, steps, max_z, min_z, value):
        discrimination = abs(max_z - min_z) / steps
        z = np.arange(min_z, max_z, discrimination)
        value = np.ones(steps) * value
        return np.array(sorted(z, reverse=True)), value
    
    @pytest.mark.unittest
    def test_classification_get_2d_slice(self):
        # extract string data
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "data_lithology.pkl")[0]
        )
        with open(input_files, "rb") as f:
            cpts = pickle.load(f)
        # set data into database
        cpts_list = []
        for name, item in cpts.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["lithology"], label="lithology"),
                    ],
                )
            )
        # run test
        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        points_2d_slice, results_2d_slice = interpolator.get_2d_slice_extra(
            location_1=cpts_list[0].location,
            location_2=cpts_list[-1].location,
            data=cpts_list,
            interpolate_variable="lithology",
            number_of_points=100,
            number_of_independent_variable_points=120,
        )
        assert len(results_2d_slice) == 100
        assert isinstance(results_2d_slice[0][0], str)
        le = preprocessing.LabelEncoder()
        le.fit(['Not available', 'Clay', 'Peat', 'Sand'])
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection="3d")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # for counter, points in enumerate(points_2d_slice):
        #     values = results_2d_slice[counter]
        #     ax.scatter(
        #         np.array(points).T[0],
        #         np.array(points).T[1],
        #         np.array(points).T[2],
        #         c=le.transform(values),
        #     )
        # plt.show()


    @pytest.mark.intergrationtest
    def test_get_2d_slice(self):
        # create two fake cpts with sparse data
        depth_1, value_1 = self.generate_fake_cpt(100, 0, -8, 10)
        depth_2, value_2 = self.generate_fake_cpt(50, -0.5, -12, 5)
        depth_3, value_3 = self.generate_fake_cpt(40, -1, -5, 1)
        location_1 = Geometry(x=0, y=0, z=0)
        cpt_1 = Data(
            location=location_1,
            independent_variable=Variable(label="depth", value=depth_1),
            variables=[Variable(label="test_value", value=value_1)],
        )
        location_2 = Geometry(x=2, y=2, z=0)
        cpt_2 = Data(
            location=location_2,
            independent_variable=Variable(label="depth", value=depth_2),
            variables=[Variable(label="test_value", value=value_2)],
        )
        location_3 = Geometry(x=4, y=4, z=0)
        cpt_3 = Data(
            location=location_3,
            independent_variable=Variable(label="depth", value=depth_3),
            variables=[Variable(label="test_value", value=value_3)],
        )
        method = InverseDistancePerDepth(nb_near_points=2)
        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        points_2d_slice, results_2d_slice, variance = interpolator.get_2d_slice_per_depth_inverse_distance(
            location_1=location_1,
            location_2=location_3,
            data=[cpt_1, cpt_2, cpt_3, cpt_1, cpt_2, cpt_3],
            interpolate_variable="test_value",
            number_of_points=100,
            number_of_independent_variable_points=120,
            interpolation_method = method
        )
        # get result
        points_2d_slice = np.reshape(np.array(points_2d_slice),
                                     (np.array(points_2d_slice).shape[0] * np.array(points_2d_slice).shape[1],
                                      3))
        results_2d_slice = np.reshape(np.array(results_2d_slice),
                                        (np.array(results_2d_slice).shape[0] * np.array(results_2d_slice).shape[1]))
        # reshape the variance
        variance = np.reshape(np.array(variance),
                                (np.array(variance).shape[0] * np.array(variance).shape[1]))
        # get results for point [1, 1, :]
        test_point_1_results = [results_2d_slice[counter]
                                for counter, point in enumerate(points_2d_slice)
                                if math.isclose(point[0], 1, abs_tol=0.001) and math.isclose(point[1], 1, abs_tol=0.001)]
        assert np.all(np.isclose(test_point_1_results[:94], 7.5, atol=0.1))
        # get results for point [3, 3, :]
        test_point_2_results = [results_2d_slice[counter]
                                for counter, point in enumerate(points_2d_slice)
                                if math.isclose(point[0], 3, abs_tol=0.001) and math.isclose(point[1], 3, abs_tol=0.001)]
        assert np.all(np.isclose(test_point_2_results[4:64], 3, atol=0.1))

    @pytest.mark.intergrationtest
    def test_get_2d_slice_inverse_dist(self):
        # create two fake cpts with sparse data
        depth_1, value_1 = self.generate_fake_cpt(100, 0, -8, 10)
        depth_2, value_2 = self.generate_fake_cpt(50, -0.5, -12, 5)
        depth_3, value_3 = self.generate_fake_cpt(40, -1, -5, 1)
        location_1 = Geometry(x=0, y=1, z=0)
        cpt_1 = Data(
            location=location_1,
            independent_variable=Variable(label="depth", value=depth_1),
            variables=[Variable(label="test_value", value=value_1)],
        )
        location_2 = Geometry(x=2, y=2, z=0)
        cpt_2 = Data(
            location=location_2,
            independent_variable=Variable(label="depth", value=depth_2),
            variables=[Variable(label="test_value", value=value_2)],
        )
        location_3 = Geometry(x=4, y=3, z=0)
        cpt_3 = Data(
            location=location_3,
            independent_variable=Variable(label="depth", value=depth_3),
            variables=[Variable(label="test_value", value=value_3)],
        )
        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        points_2d_slice, results_2d_slice = interpolator.get_2d_slice_extra(
            location_1=location_1,
            location_2=location_3,
            data=[cpt_1, cpt_2, cpt_3],
            interpolate_variable="test_value",
            number_of_points=100,
            number_of_independent_variable_points=120,
        )


        assert len(points_2d_slice) == 100
        assert len(points_2d_slice[0]) == 120
        assert len(points_2d_slice[0][0]) == 3
        assert len(results_2d_slice) == 100
        assert len(results_2d_slice[0]) == 120


    @pytest.mark.intergrationtest
    def test_get_2d_slice_real_data(self):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "test_case_DF.pickle")[0]
        )
        with open(input_files, "rb") as f:
            (cpts, resistivity, insar) = pickle.load(f)
        # create List[Data]
        cpts_list = []
        for name, item in cpts.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["water"], label="water"),
                        Variable(value=item["tip"], label="tip"),
                        Variable(value=item["IC"], label="IC"),
                        Variable(value=item["friction"], label="friction"),
                    ],
                )
            )
        location_1 = Geometry(x=64348.1, y=393995.8, z=0)
        location_2 = Geometry(x=64663.8, y=393960.3, z=0)

        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        points_2d_slice, results_2d_slice = interpolator.get_2d_slice_extra(
            location_1=location_1,
            location_2=location_2,
            data=cpts_list,
            interpolate_variable="IC",
            number_of_points=100,
            number_of_independent_variable_points=120,
        )
        assert len(points_2d_slice) == 98
        assert len(points_2d_slice[0][0]) == 3
        assert len(results_2d_slice) == 98
        assert len(results_2d_slice[0]) == 120
        assert len(results_2d_slice[0]) == 120

    @pytest.mark.intergrationtest
    def test_get_2d_slice_same_x(self):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "test_case_DF.pickle")[0]
        )
        with open(input_files, "rb") as f:
            (cpts, resistivity, insar) = pickle.load(f)
        # create List[Data]
        cpts_list = []
        for name, item in cpts.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["water"], label="water"),
                        Variable(value=item["tip"], label="tip"),
                        Variable(value=item["IC"], label="IC"),
                        Variable(value=item["friction"], label="friction"),
                    ],
                )
            )
        location_1 = Geometry(x=64348, y=393995, z=0)
        location_2 = Geometry(x=64348, y=393860, z=0)
        interpolator_slice = Nearest()

        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        points_2d_slice, results_2d_slice = interpolator.get_2d_slice_extra(
            location_1=location_1,
            location_2=location_2,
            data=cpts_list,
            interpolate_variable="IC",
            number_of_points=100,
            number_of_independent_variable_points=120,
        )

        assert len(points_2d_slice) == 99
        assert len(points_2d_slice[0][0]) == 3
        assert len(results_2d_slice) == 99
        assert len(results_2d_slice[0]) == 120
        assert len(results_2d_slice[0]) == 120
        assert np.array(points_2d_slice)[:, :, 0].min() == location_1.x
        assert np.array(points_2d_slice)[:, :, 0].max() == location_1.x

    @pytest.mark.intergrationtest
    def test_get_2d_slice_same_y(self):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "test_case_DF.pickle")[0]
        )
        with open(input_files, "rb") as f:
            (cpts, resistivity, insar) = pickle.load(f)
        # create List[Data]
        cpts_list = []
        for name, item in cpts.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["water"], label="water"),
                        Variable(value=item["tip"], label="tip"),
                        Variable(value=item["IC"], label="IC"),
                        Variable(value=item["friction"], label="friction"),
                    ],
                )
            )
        location_1 = Geometry(x=64348.1, y=393995, z=0)
        location_2 = Geometry(x=64663.8, y=393995, z=0)
        interpolator_slice = Nearest()

        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        points_2d_slice, results_2d_slice = interpolator.get_2d_slice_extra(
            location_1=location_1,
            location_2=location_2,
            data=cpts_list,
            interpolate_variable="IC",
            number_of_points=100,
            number_of_independent_variable_points=120,
        )
        assert len(points_2d_slice) == 97
        assert len(points_2d_slice[0][0]) == 3
        assert len(results_2d_slice) == 97
        assert len(results_2d_slice[0]) == 120

    @pytest.mark.intergrationtest
    def test_get_user_defined_surface(self):
        # get ahn line
        spacial_utils = SpatialUtils()
        surface_line = []
        for i in np.arange(64358.1, 64443.8, 1):
            surface_line.append([i, 393995])
        spacial_utils.get_ahn_surface_line(np.array(surface_line))

        location_1 = Geometry(x=64358.1, y=393995, z=0)
        location_2 = Geometry(x=64443.8, y=393995, z=0)
        interpolator_slice = Nearest()

        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        closest_top = interpolator.get_user_defined_surface(
            location_1=location_1,
            location_2=location_2,
            number_of_points=10,
            surface=spacial_utils.AHN_data,
        )

    @pytest.mark.intergrationtest
    def test_get_2d_with_user_defined_line(self):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "test_case_DF.pickle")[0]
        )
        with open(input_files, "rb") as f:
            (cpts, resistivity, insar) = pickle.load(f)
        # create List[Data]
        cpts_list = []
        for name, item in cpts.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["water"], label="water"),
                        Variable(value=item["tip"], label="tip"),
                        Variable(value=item["IC"], label="IC"),
                        Variable(value=item["friction"], label="friction"),
                    ],
                )
            )
        # get ahn line
        spacial_utils = SpatialUtils()
        surface_line = []
        for i in np.arange(64358.1, 64443.8, 1):
            surface_line.append([i, 393995])
        spacial_utils.get_ahn_surface_line(np.array(surface_line))

        location_1 = Geometry(x=64358.1, y=393995, z=0)
        location_2 = Geometry(x=64443.8, y=393995, z=0)
        interpolator_slice = Nearest()

        interpolator = Interpolate2DSlice(interpolation_method_surface=InverseDistance(nb_near_points=2))
        points_2d_slice, results_2d_slice = interpolator.get_2d_slice_extra(
            location_1=location_1,
            location_2=location_2,
            data=cpts_list,
            interpolate_variable="IC",
            number_of_points=100,
            number_of_independent_variable_points=120,
            #top_surface=spacial_utils.AHN_data,
            #bottom_surface=np.array(
            #    [[location_1.x, location_1.y, -10], [location_2.x, location_2.y, -10]]
            #),
        )

        assert len(results_2d_slice[0]) == 120
        assert round(np.array(points_2d_slice)[:, :, 1].min()) == location_1.y
        assert round(np.array(points_2d_slice)[:, :, 1].max()) == location_1.y
