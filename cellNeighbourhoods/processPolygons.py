from shapely.geometry import Polygon
import itertools
from scipy.spatial.distance import euclidean
import math
import multiprocessing as mp
import streamlit as st

class makePolygons():
    def __init__(self,
                 dict_shapes: dict,
                 verbose: int = 0):
        self._dict_shapes : dict = dict_shapes
        self._dict_polygons : dict = {}
        
        self._convert_shapes_to_polygons()
        
    def _convert_shapes_to_polygons(self):
        self._dict_polygons = {shape_id: {'polygon': Polygon(
            coords)} for shape_id, coords in self._dict_shapes.items()}
    
    @staticmethod
    def major_axis(polygon: Polygon):
        # Extract the coordinates of the polygon's exterior plus interior(s) if they exist
        vertices = list(polygon.exterior.coords) + [pt for interior in polygon.interiors for pt in interior.coords]

        # Calculate distances between every pair of points
        distances = [euclidean(pt1, pt2) for pt1, pt2 in itertools.combinations(vertices, 2)]

        # The major axis length is the maximum distance found
        return max(distances)
    
    @staticmethod
    def calculate_minor_axis_length(polygon):
        # Calculate the minimum rotated rectangle (MRR)
        mrr = polygon.minimum_rotated_rectangle
        
        # Extract the coordinates of the MRR
        mrr_coords = list(mrr.exterior.coords)
        
        # Calculate the lengths of the sides of the MRR
        # Since MRR is a rectangle, we can just calculate two adjacent sides
        side1_length = math.sqrt((mrr_coords[0][0] - mrr_coords[1][0])**2 + (mrr_coords[0][1] - mrr_coords[1][1])**2)
        side2_length = math.sqrt((mrr_coords[1][0] - mrr_coords[2][0])**2 + (mrr_coords[1][1] - mrr_coords[2][1])**2)
        
        # The minor axis length is the smaller of the two sides
        return min(side1_length, side2_length)
    
    @staticmethod
    def circularity(polygon):
        # Calculate area of the polygon
        area = polygon.area

        # Calculate perimeter of the polygon
        perimeter = polygon.length

        # Calculate circularity
        return (4 * math.pi * area) / (perimeter ** 2)
    
    @staticmethod
    def convexity(polygon):
        # Calculate the area of the polygon
        polygon_area = polygon.area

        # Calculate the convex hull of the polygon and then its area
        convex_hull_area = polygon.convex_hull.area

        # Calculate convexity
        return polygon_area / convex_hull_area
    
    # @property
    # def get_properties(self):
    #     # ! Time consuming step
        
        # for shape_id, dict_properties in self._dict_polygons.items():
        #     dict_properties['area'] = dict_properties['polygon'].area
        #     dict_properties['centroid'] = dict_properties['polygon'].centroid.coords[0]
        #     # dict_properties['major_axis'] = makePolygons.major_axis(dict_properties['polygon'])
        #     # dict_properties['minor_axis'] = makePolygons.calculate_minor_axis_length(dict_properties['polygon'])
        #     dict_properties['circularity'] = makePolygons.circularity(dict_properties['polygon'])
        #     dict_properties['convexity'] = makePolygons.convexity(dict_properties['polygon'])
            
        #     self._dict_polygons[shape_id].update(dict_properties)
        # return self._dict_polygons
        
    @staticmethod
    def _calculate_properties(shape_id, dict_properties):
        dict_properties['area'] = dict_properties['polygon'].area
        dict_properties['centroid'] = dict_properties['polygon'].centroid.coords[0]
        # dict_properties['major_axis'] = makePolygons.major_axis(dict_properties['polygon'])
        # dict_properties['minor_axis'] = makePolygons.calculate_minor_axis_length(dict_properties['polygon'])
        dict_properties['circularity'] = makePolygons.circularity(
            dict_properties['polygon'])
        dict_properties['convexity'] = makePolygons.convexity(
            dict_properties['polygon'])

        return shape_id, dict_properties

    def _parallelize_properties_calculation(self):
        pool = mp.Pool(mp.cpu_count() - 1)
        results = []
        
        for shape_id, dict_properties in self._dict_polygons.items():
            result = pool.apply_async(self._calculate_properties, (shape_id, dict_properties))
            results.append(result)

        pool.close()
        pool.join()

        for result in results:
            shape_id, dict_properties = result.get()
            self._dict_polygons[shape_id].update(dict_properties)

    @property
    def get_properties(self):
        self._parallelize_properties_calculation()
        return self._dict_polygons
    

def get_polygon_metric(dict_polygons):
    lst_area = []
    lst_circularity = []
    lst_convexity = []

    for _, dict_properties in dict_polygons.items():
        lst_area.append(dict_properties['area'])
        lst_circularity.append(dict_properties['circularity'])
        lst_convexity.append(dict_properties['convexity'])

    return lst_area, lst_circularity, lst_convexity


def make_polygons(dict_shapes: dict, bool_polygon_stats: bool =True):
    st.toast('Converting shapes into polygons.')
    my_polygons = makePolygons(dict_shapes, verbose=0)
    dict_polygons = my_polygons.get_properties
    
    if bool_polygon_stats == False:
        return dict_polygons

    st.toast('Calculating properties of polygons.')
    lst_area, lst_circularity, lst_convexity = get_polygon_metric(dict_polygons)
    return dict_polygons, lst_area, lst_circularity, lst_convexity