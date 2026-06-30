import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from parseXML import read_from_xml, write_to_xml
from processPolygons import make_polygons, get_polygon_metric
from processCentroids import CentroidSelector
from collections import defaultdict
from classifyPolygons import classifyPolygons

st.set_page_config(layout="wide")

# initialize a session state
if 'cl_centroids' not in st.session_state:
    st.session_state['cl_centroids'] = defaultdict(list)

dict_params = dict()

def get_uploaded_file() -> tuple:
    col0, col1 = st.columns([0.75, 0.25])
    with col0:
        uploaded_file = st.file_uploader(
            "XML file", type="xml", accept_multiple_files=False)

    with col1:
        n_coords_min = int(st.number_input(
            'Minimum number of coordinates per shape:', 4, value=4))
        
    return uploaded_file, n_coords_min

def main():
    st.title("Find shape neighbourhoods using supervised centroids.")
    
    # Get uploaded file and the minimum number if coordinates.
    uploaded_fobj, n_coords_min  = get_uploaded_file()
    
    # Terminate execution if the uploaded file is empty.
    if uploaded_fobj is None:
        return None
    print('\n\n====================================')
    print(f'Uploaded file: {uploaded_fobj.name}')
    
    # Read in the xml file
    with st.spinner('Reading XML file...'):
        dict_shapes = read_from_xml(uploaded_fobj, n_coords_min)
        print(f'- Uploaded file has {len(dict_shapes)} shapes.')
    
    # Convert shapes into polygons  
    with st.spinner('Converting shapes into polygons...'):
        dict_polygons, lst_area, lst_circularity, lst_convexity = make_polygons(dict_shapes)
        print(f'- Uploaded file has {len(dict_polygons)} polygons.')
        
        # Visualize the output of polygons
        st.markdown('## Shape overviews')
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        ax[0].hist(np.log10(lst_area), bins=51)
        ax[0].set_xlabel('Area (log10)')

        ax[1].hist(np.array(lst_circularity), bins=51)
        ax[1].set_xlabel('Circularity')

        ax[2].hist(np.array(lst_convexity), bins=51)
        ax[2].set_xlabel('Convexity')
        fig.tight_layout()
        st.pyplot(fig)
    
    # Parameters for filtering and clustering
    st.markdown('## Filtering and clustering settings')
    col0, col1, col2 = st.columns([0.5, 0.25, 0.25])
    with col0:
        st.markdown('### Polygon filtering')
        dict_params['polygon_area'] = st.slider('Select a range of values for the area:',
                                min(np.log10(lst_area)), max(np.log10(lst_area)),
                                (min(np.log10(lst_area)), max(np.log10(lst_area))))

        dict_params['polygon_circ'] = st.slider('Select a range of values for the circularity:',
                                min(np.array(lst_circularity)), 1.0,
                                (min(np.array(lst_circularity)), 1.0))

        dict_params['polygon_convex'] = st.slider('Select a range of values for the convexity:',
                                min(np.array(lst_convexity)), 1.0,
                                (min(np.array(lst_convexity)), 1.0))
    
    with col1:
        st.markdown('### Cluster requirements')
        dict_params['n_clusters'] = int(st.number_input('Number of clusters.', 1, value=14))
        temp_shapes_per_neigh = int(st.number_input('Number of shapes in each cluster.', 1, value=800))
        temp_rel_threshold = float(st.number_input('Relative range for number of shapes. (Percentage between 0 and 100)', 0.0, 100.0, value=10.0)) / 100
        
        dict_params['shapes_per_neigh'] = [int(temp_shapes_per_neigh - (temp_shapes_per_neigh * temp_rel_threshold)),
                                            int(temp_shapes_per_neigh + (temp_shapes_per_neigh * temp_rel_threshold))]

    with col2:
        st.markdown('### Clustering parameters')
        dict_params['epsilon_in'] = int(st.number_input('Epsilon', 1,
                                                        value=1500,
                                                        help='Starting point for the epsilon sweep. Smaller value increases computation time.'))
    
    print()
    print('- Clustering settings.')
    for key, value in dict_params.items():
        print(f' - {key}: {value}')
    
    # Filter polygons
    st.toast('Filtering polygons.')
    print()
    print('- Filtering polygons')
    my_classifier = classifyPolygons(dict_polygons)
    
    str_0 = my_classifier.filter_shapes(
        'area', 10**dict_params['polygon_area'][0]-1, 10**dict_params['polygon_area'][1]+1)
    st.info(str_0)

    str_1 = my_classifier.filter_shapes(
        'circularity', dict_params['polygon_circ'][0]-0.01, dict_params['polygon_circ'][1]+0.01)
    st.info(str_1)

    str_2 = my_classifier.filter_shapes(
        'convexity', dict_params['polygon_convex'][0]-0.01, dict_params['polygon_convex'][1]+0.01)
    st.info(str_2)
    
    print(f' - Polygons before and after filtering: {len(dict_polygons)}', end=' -> ')
    dict_polygons = my_classifier.get_filtered_polygon(dict_polygons)
    print(len(dict_polygons))
    
    # Centroid processing
    st.markdown('## Centroid selection')
    centroid_processor = CentroidSelector(dict_polygons, st.session_state)
    col0, col1 = st.columns([0.75, 0.25])
    
    with col1:
        centroid_processor.text_centroids()
        st.button('Clear', on_click=centroid_processor.reset_cl_centroid)
    
    with col0:
        centroid_processor.render_centroids()
    
    dict_centroids = centroid_processor.get_centroids
    
    if len(dict_centroids['x']) == 0:
        return False    
    
    if st.button('Process'):
        # process centroids
        with st.spinner('Clustering polygons...'):
            arr_centroid = np.array([dict_centroids['x'], dict_centroids['y']]).T
            
            df_data_clustered, fig = my_classifier.make_clusters(arr_centroid,
                                        dict_params['epsilon_in'],
                                        *dict_params['shapes_per_neigh'])
            
            print(df_data_clustered)
            col0, col1 = st.columns([0.75, 0.25])
            with col0:
                st.pyplot(fig)

            if df_data_clustered is not None:
                st.balloons()
        
        with st.spinner('Preparing XML files...'):
            write_to_xml(df_data_clustered, uploaded_fobj, fig)
        
if __name__ == "__main__":
    main()
