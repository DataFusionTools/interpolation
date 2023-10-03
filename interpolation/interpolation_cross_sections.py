import shapefile
from shapely import ops
from shapely.geometry import LineString, Point, MultiLineString
import numpy as np

def get_cross_sections_along_line(shapefile_embankment,
                                  inter_cross_section_distance: float,
                                  cross_section_length: float):
    """ Functions that defines the RDx and RDy coordinates of the beginning and end point of cross-sections
    perpendicular to the embankment. The beginning of the cross-section will be left of the embankment line. The start
    point of the line element in the shapefile determines what is left.

    :param shapefile_embankment: The shapefile (*.shp) of the embankment line, line element. Usually the crest of the
                                 embankment.
    :param inter_cross_section_distance: The distance between the consecutive cross-sections along the embankment.
    :param cross_section_length: The length of each cross-section. Half of this length will be left of the embankment
                                 line, the other half right.

    :returns: A list of dictionaries. In each dictionary there are the RDx and RDy of the beginning, x_start and
              y_start, and end point, x_end and y_end, of a single cross-section.

    """

    # Read the embankment line from shapefile into shapely LineString

    shape = shapefile.Reader(shapefile_embankment)

    line = []

    for i in range(len(shape.shapeRecords())):
        feature = shape.shapeRecords()[i]
        first = feature.shape.__geo_interface__
        line.append(LineString([Point(i[0], i[1]) for i in first['coordinates']]))

    line = MultiLineString(line)
    line = ops.linemerge(line)

    # Define the locations of the cross-sections along the embankment line

    points = [line.interpolate(dist) for dist in np.arange(0, line.length, inter_cross_section_distance)]

    # Define the orientation of the embankment line at those locations

    _offsetpts = [line.interpolate(currDist + 0.001) for currDist in np.arange(0, line.length, inter_cross_section_distance)]

    orientations_embankment = [np.arctan(((points[idx].coords[0][1] - _offsetpts[idx].coords[0][1]) /
                                         (points[idx].coords[0][0] - _offsetpts[idx].coords[0][0])))
                               for idx, pt in enumerate(_offsetpts)]

    # Define the start and end points of the cross-sections given their location and orientation

    profiles = []

    for point, orientation_embankment in zip(points, orientations_embankment):

        profiles.append({'x_start': point.xy[0][0] + cross_section_length/2 * np.cos(orientation_embankment + np.pi / 2),
                         'x_end':   point.xy[0][0] - cross_section_length/2 * np.cos(orientation_embankment + np.pi / 2),
                         'y_start': point.xy[1][0] + cross_section_length/2 * np.sin(orientation_embankment + np.pi / 2),
                         'y_end':   point.xy[1][0] - cross_section_length/2 * np.sin(orientation_embankment + np.pi / 2)})

    return profiles, line
