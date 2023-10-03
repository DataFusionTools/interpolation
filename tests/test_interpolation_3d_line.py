import pytest
import shapefile
import pickle

from core.data_input import Data, Variable
from interpolation.interpolation_3d_line import Interpolate3DLine
from utils import TestUtils
from core.data_input import Geometry


class TestInterpolate3dLine:

    @pytest.mark.intergrationtest
    def test_get_3d_line(self):
        name_shapefile = "VEILIGHEID_DIJKTRAJECTEN_PRIMAIRE_WATERKERINGEN.shp"
        dijk_traject_file = str(
            TestUtils.get_test_files_from_local_test_dir("VEILIGHEID_DIJKTRAJECTEN_PRIMAIRE_WATERKERINGEN", name_shapefile)[0]
        )
        shape = shapefile.Reader(dijk_traject_file)
        feature = shape.shapeRecords()[197]

        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "cpts_extent.pickle")[0]
        )
        with open(input_files, "rb") as f:
            cpts = pickle.load(f)
        # create List[Data]
        cpts_list = []
        for item in cpts:
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=-1 * item["depth"], label="depth"),
                    variables=[
                        Variable(value=item["water"], label="water"),
                        Variable(value=item["tip"], label="tip"),
                        Variable(value=item["IC"], label="IC"),
                        Variable(value=item["friction"], label="friction"),
                    ],
                )
            )
        interpolate = Interpolate3DLine()
        points, value =  interpolate.get_3d_line(feature.shape.points[:10], cpts_list, "IC", 20, -10)
        assert len(value) == 200
              








        
   