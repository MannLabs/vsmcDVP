import numpy as np
import streamlit as st
import colorcet as cc
import matplotlib.pyplot as plt

class CentroidSelector():
    def __init__(self,
                 dict_polygons: dict,
                 st_state: dict):
        self._dict_polygons = dict_polygons
        self._st_state = st_state
        self._dict_polygon_centroids: dict = self._get_polygon_centroids()
        self._dict_cluster_centroids: dict = self._st_state['cl_centroids']
        
        self._lst_centroids: list = []
        
    def _get_polygon_centroids(self) -> dict[str, list]:
        lst_cent_x = []
        lst_cent_y = []
        lst_color = []
        lst_size = []
        lst_label = []

        for _, dict_data in self._dict_polygons.items():
            x, y = dict_data['centroid']
            lst_cent_x.append(x)
            lst_cent_y.append(y)
            lst_color.append('#fcbba1')
            lst_size.append(1.0)
            lst_label.append(None)

        return {'x': lst_cent_x.copy(),
                                        'y': lst_cent_y.copy(),
                                        'color': lst_color.copy(),
                                        'size': lst_size.copy(),
                                        'label': lst_label.copy()}

    def render_centroids(self):
        st.markdown('### Plaque map')

        lst_cent_idx = []
        lst_cent_x = []
        lst_cent_y = []
        lst_color = []
        lst_size = []
        lst_label = []
        
        # Merge two dictionaries
        for dict_data in [self._dict_polygon_centroids,
                          self._dict_cluster_centroids]:
            lst_cent_idx.extend(dict_data.get('idx', [None]*len(dict_data['x'])))
            lst_cent_x.extend(dict_data['x'])
            lst_cent_y.extend(dict_data['y'])
            lst_color.extend(dict_data['color'])
            lst_size.extend(dict_data['size'])
            lst_label.extend(dict_data['label'])
            
        
        # Visualize the tissue centroids
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.grid(color='#41b6c4', linestyle='dotted', linewidth=0.5)
        ax.scatter(lst_cent_x, lst_cent_y,
                   c=lst_color, s=lst_size)
        
        # Add cluster label
        for x, y, t in zip(lst_cent_x, lst_cent_y, lst_cent_idx):
            if t is None:
                continue
            ax.text(x, y, t, ha='center', va='center', color='#d9d9d9')
        
        ax.ticklabel_format(useOffset=False, style='plain')
        fig.tight_layout()
        st.pyplot(fig)
        
    def _update_cl_centroid(self, idx, x_coord, y_coord):
        cl_idx = len(self._dict_cluster_centroids['color'])
        
        self._dict_cluster_centroids['idx'].append(idx)
        self._dict_cluster_centroids['x'].append(int(x_coord))
        self._dict_cluster_centroids['y'].append(int(y_coord))
        # self._dict_cluster_centroids['color'].append(cc.glasbey[cl_idx])
        self._dict_cluster_centroids['color'].append('#525252')
        self._dict_cluster_centroids['size'].append(150.0)
        self._dict_cluster_centroids['label'].append(str(cl_idx))
        
        print(f'Added centroid index: {cl_idx}')
        
    def reset_cl_centroid(self):
        self._dict_cluster_centroids['idx'] = []
        self._dict_cluster_centroids['x'] = []
        self._dict_cluster_centroids['y'] = []
        self._dict_cluster_centroids['color'] = []
        self._dict_cluster_centroids['size'] = []
        self._dict_cluster_centroids['label'] = []
    
    def text_centroids(self):
        st.markdown('### Centroid coordinates')
        str_centroids = st.text_area('IDX, XXXX, YYYY', height=750, placeholder='#, #####, #####')
        
        # if len(self._dict_cluster_centroids['x']) == 0:
        #     str_centroids = st.text_area('IDX, XXXX, YYYY', height=750, placeholder='#, #####, #####')
        # else:
        #     str_value = ''
        #     for idx, (x, y) in enumerate(zip(self._dict_cluster_centroids['x'], self._dict_cluster_centroids['y'])):
        #         str_value = str_value + f'{idx}, {x}, {y}\n'

        #     str_centroids = st.text_area('Centroids', str_value, height=750)
        
        self.reset_cl_centroid()
        
        if len(str_centroids) == 0:
            return None
        
        for centroid in str_centroids.split('\n'):
            lst_centroid = centroid.split(',')
            if len(lst_centroid) > 1:
                lst_centroid = [x.strip() for x in lst_centroid]
                idx = lst_centroid[0]
                x = lst_centroid[1]
                y = lst_centroid[2]
                
                self._update_cl_centroid(idx, x, y)
    
    @property
    def get_centroids(self):
        return self._dict_cluster_centroids