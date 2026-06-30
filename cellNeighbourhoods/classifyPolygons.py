from sklearn.cluster import HDBSCAN, BisectingKMeans, KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import colorcet as cc
from matplotlib.colors import ListedColormap
from constants import lst_words

np.random.seed(42)
np.random.shuffle(lst_words)

class classifyPolygons:
    def __init__(self, dict_polygons:dict):
        self._dict_polygons = dict_polygons
        self._df_polygons_ori: 'pd.DataFrame'
        self._df_polygons_filt: 'pd.DataFrame'
        self._df_centroids: 'pd.DataFrame'
        self._convert2df()
        
    def _convert2df(self) -> 'pd.DataFrame':
        dict_data = defaultdict(list)
        
        for shape_id, dict_properties in self._dict_polygons.items():
            dict_data['shape_id'].append(shape_id)
            dict_data['area'].append(dict_properties['area'])
            dict_data['centroid_x'].append(dict_properties['centroid'][0])
            dict_data['centroid_y'].append(dict_properties['centroid'][1])
            
            # dict_data['major_axis'].append(dict_properties['major_axis'])
            # dict_data['minor_axis'].append(dict_properties['minor_axis'])
            
            dict_data['circularity'].append(dict_properties['circularity'])
            dict_data['convexity'].append(dict_properties['convexity'])
            
        self._df_polygons_ori = pd.DataFrame(dict_data)
        self._df_polygons_ori.set_index('shape_id', inplace=True)
        
        self._df_polygons_filt = pd.DataFrame(dict_data).copy()
                
    def filter_shapes(self,
                      metric: str,
                      min_val: float,
                      max_val: float,
                      percentile:bool = False):
        
        if metric not in ['area', 'circularity', 'convexity']:
            print('Incorrect metric, must be one of: `area`, `circularity`, `convexity`')
            return None
        
        if percentile:
            min_val = np.percentile(self._df_polygons_filt[metric], min_val)
            max_val = np.percentile(self._df_polygons_filt[metric], max_val)
        
        str_out = f' - Filtering by `{metric}`: {len(self._df_polygons_filt)} -> '
        
        self._df_polygons_filt = self._df_polygons_filt.loc[self._df_polygons_filt[metric] >= min_val, :]
        self._df_polygons_filt = self._df_polygons_filt.loc[self._df_polygons_filt[metric] <= max_val, :]
        str_out += f'{len(self._df_polygons_filt)}.'
        
        print(str_out)
        return str_out
    
    def get_filtered_polygon(self, dict_polygons:dict):
        lst_shape_id = self._df_polygons_filt['shape_id'].to_list()
        dict_out = {x:dict_polygons[x] for x in dict_polygons.keys() if x in lst_shape_id}
        
        temp = len(dict_polygons) - len(dict_out)
        return dict_out
    
    def initial_clusters(self,
                         n_clusters: int):
        # Create the dataframe
        df_data = self._df_polygons_filt.loc[:, ['centroid_x', 'centroid_y']].copy()
        xy_data = df_data.copy().values

        # * 1. Determine centroids using hdbscan
        # TODO: Replace this with an optimizer?
        for epsilon in range(300, 15100, 100):
            cluster_labels, cluster_centroids = self.cluster_hdbscan(xy_data, cluster_epsilon=epsilon)
            if cluster_centroids.shape[0] <= n_clusters+2:
                break

        print(f'Found {cluster_centroids.shape[0]} clusters at {epsilon} epsilon.')
        
        return cluster_centroids
    
    def make_clusters(self,
                      n_clusters: int = 17,
                      epsilon_in: float = 500,
                      shapes_per_neigh_min: int = 450,
                      shapes_per_neigh_max: int = 550):
        # Create the dataframe
        df_data = self._df_polygons_filt.loc[:, ['centroid_x', 'centroid_y']].copy()
        xy_data = df_data.copy().values
        
        # * 1. Determine centroids using hdbscan
        # TODO: Replace this with an optimizer?

        if isinstance(n_clusters, int):
            for epsilon in range(300, 15100, 100):
                cluster_labels, cluster_centroids = self.cluster_hdbscan(xy_data, cluster_epsilon=epsilon)
                if cluster_centroids.shape[0] <= n_clusters+2:
                    break
            
            print(f'Found {cluster_centroids.shape[0]} clusters at {epsilon} epsilon.')
        
        elif isinstance(n_clusters, np.ndarray):
            print('Using pre-determined centroids!')
            cluster_centroids = n_clusters
        else:
            assert False, "It shouldn't be here"
        
        # * 2. Assign shapes to a hard clusters using KMeans
        # ! Working here!
        km_panels = self.cluster_euc(xy_data, cluster_centroids)
        # km_panels, _ = self.cluster_kmeans(xy_data,
        #                                    n_clusters=cluster_centroids.shape[0],
        #                                    init=cluster_centroids)
        
        df_data['km_cl'] = km_panels
        for idx, (_, df_group) in enumerate(df_data.groupby('km_cl')):
            plt.scatter(df_group['centroid_x'], df_group['centroid_y'],
                        s=0.2,
                        c=cc.glasbey[idx],
                        marker='o')
        plt.savefig('XX_figure_clustered_kmeans.png', dpi=400)
        plt.close()
        
        # * 3. Inside each k mean cluster, define clusters based on density (hdbscan)
        dict_centroids = {}
        lst_dfs = []
        
        for km_cl, df_km_cluster in df_data.groupby('km_cl'):
            # Perform clustering with varying thresholds until shape counts are reached
            # TODO: Replace this with an optimizer?
            for cluster_epsilon in range(epsilon_in, 25_000, 100):
                xy_data = df_km_cluster.loc[:, ['centroid_x', 'centroid_y']].copy().values
                cluster_labels, cluster_centroids = self.cluster_hdbscan(xy_data, cluster_epsilon=cluster_epsilon)
                # cluster_labels, cluster_centroids = self.cluster_random(xy_data, max_k_size=shapes_per_neigh)
                
                k_counter = Counter(cluster_labels[cluster_labels >= 0])
                k_counter = {k: size for k, size in k_counter.most_common(1) if size >= shapes_per_neigh_min}
                
                if len(k_counter) > 0:
                    print(f'Converged with {len(k_counter)} clusters and DBSCAN epsilon: {cluster_epsilon}.')
                    break
            else:
                print(f'Did not converge! epsilon: {cluster_epsilon}.')
            
            df_km_cluster['shape_cl'] = cluster_labels

            # Filter dataframe to only keep the most common clusters
            lst_shape_cls = list(k_counter.keys())
            df_km_cluster = df_km_cluster.loc[[x in lst_shape_cls for x in df_km_cluster['shape_cl']], :]
            
            # Only keep the first `shapes_per_neigh` shapes
            dict_centroids.update({(km_cl, shape_cl): [centroid[0], centroid[1], shapes_per_neigh_max if k_counter[shape_cl] > shapes_per_neigh_max else k_counter[shape_cl]]
                                  for shape_cl, centroid in enumerate(cluster_centroids) if shape_cl in lst_shape_cls})
            
            try:
                lst_df_temp = []
                for _, df_temp in df_km_cluster.groupby('shape_cl'):
                    df_temp = df_temp.iloc[:shapes_per_neigh_max, :].copy()
                    lst_df_temp.append(df_temp)
                df_km_cluster = pd.concat(lst_df_temp, axis=0)
            except ValueError:
                print('-- No shapes to keep. Shape', df_km_cluster.shape)
            else:
                dict_centroids.update({(km_cl, shape_cl): [centroid[0], centroid[1], shapes_per_neigh_max if k_counter[shape_cl] > shapes_per_neigh_max else k_counter[shape_cl]]
                                       for shape_cl, centroid in enumerate(cluster_centroids) if shape_cl in lst_shape_cls})
            
            lst_dfs.append(df_km_cluster)
        
        # Aggregate the dataframes into one
        df_data_clustered = pd.concat(lst_dfs, axis=0)
        df_data_clustered['final_cl'] = df_data_clustered['km_cl'].astype(str) + '_' + df_data_clustered['shape_cl'].astype(str)
        
        # Get cluster names
        dict_names = {x:lst_words[idx] for idx, x in enumerate(df_data_clustered['final_cl'].unique())}
        df_data_clustered['final_cl_name'] = [dict_names[x] for x in df_data_clustered['final_cl']]
        
        # 4. Plot the clusters
        # Plot background points
        fig, ax = plt.subplots(figsize=(7, 7))
        df_temp = df_data.loc[:, ['centroid_x', 'centroid_y', 'km_cl']].copy()
        for idx, (_, df_group) in enumerate(df_temp.groupby('km_cl')):
            ax.scatter(df_group['centroid_x'], df_group['centroid_y'],
                        s=0.15,
                        c='#bdbdbd',
                        marker='o')
            
        for idx, (_, df_group) in enumerate(df_data_clustered.groupby('final_cl')):
            ax.scatter(df_group['centroid_x'], df_group['centroid_y'],
                        s=0.2,
                        c=cc.glasbey[idx],
                        marker='o')

        
        for (km_cl, shape_cl), (centroid_x, centroid_y, shape_count) in dict_centroids.items():
            cl_name = dict_names[f'{km_cl}_{shape_cl}']
            ax.text(centroid_x, centroid_y, f'{cl_name}: ({shape_count})',
                     fontsize='x-small',
                     c='black')
        
        ax.ticklabel_format(useOffset=False, style='plain')
        # plt.savefig('XX_figure_clustered.png', dpi=400)
        # plt.close()
        
        return df_data_clustered, fig
        

    @staticmethod
    def cluster_hdbscan(arr_data,
                        min_k_size:int=20, # 20
                        min_samples:int=4,
                        cluster_epsilon:float=4500,
                        max_k_size: int = 520):
        
        # print('Performing clustering using HDBSCAN.')
        clustering = HDBSCAN(min_cluster_size=min_k_size,  # samples threshold for cluster 20
                             min_samples=min_samples,  # samples in neighborhood for core point 4
                             cluster_selection_epsilon=cluster_epsilon,  # ! Maybe the parameter?
                             max_cluster_size=max_k_size,
                             store_centers='centroid',
                             n_jobs=20)

        clustering.fit(arr_data)
        return clustering.labels_, clustering.centroids_
    
    @staticmethod
    def cluster_random(arr_data,
                       max_k_size: int = 520):
        bool_mask = np.array([-1] * arr_data.shape[0])
        bool_mask[:max_k_size] = 1
        np.random.shuffle(bool_mask)
        
        return bool_mask, []
    
    @staticmethod
    def cluster_kmeans(arr_data, n_clusters, init='k-means++'):
        # print('Performing clustering using Kmeans.')
        # clustering = BisectingKMeans(n_clusters=n_clusters,
        #                              init=init,
        #                              n_init=51,
        #                              random_state=0)
        clustering = KMeans(n_clusters=n_clusters,
                                     init=init,
                                     n_init=1,
                                     random_state=0)
        
        clustering.fit(arr_data)
        return clustering.labels_, clustering.cluster_centers_
    
    @staticmethod
    def cluster_euc(arr_data, centroids):
        lst_data = []
        
        for polygon in arr_data:
            lst_distances = []
            for centroid in centroids:
                distance = np.linalg.norm(polygon - centroid)
                lst_distances.append(distance)
            
            lst_data.append(np.argmin(lst_distances))
            
        return np.array(lst_data)
