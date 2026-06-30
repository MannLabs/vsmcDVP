import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import os, tempfile
from zipfile import ZipFile
import streamlit as st
from datetime import datetime

class parseXMLfile():
    def __init__(self,
                 file: str,
                 n_coordinates_min: int,
                 verbose: int = 0):
        
        self._input_floc: str = file
        self._verbose: int = verbose
        self._n_coordinates_min: int = n_coordinates_min
        self._dict_shapes: dict = {}
        
        self._skipped_shaped: int = 0
        
    def _read_file(self):        
        tree = ET.parse(self._input_floc)
        root = tree.getroot()
        
        # Iterate through the XML tree to find elements that represent shapes
        
        for shape in root:
            if not shape.tag.startswith('Shape_'):
                continue
            
            coordinates = []
            # The number of points in the shape
            point_count = int(shape.find('PointCount').text)
            
            for i in range(1, point_count + 1):
                x_tag = f'X_{i}'
                y_tag = f'Y_{i}'
                
                x_value = int(shape.find(x_tag).text)
                y_value = int(shape.find(y_tag).text)
                
                coordinates.append((x_value, y_value))
            
            #! Implement filter for number of coordinates in a shape
            if len(coordinates) < self._n_coordinates_min:
                self._skipped_shaped += 1
                continue
            
            self._dict_shapes[shape.tag] = coordinates

    @property
    def shapes(self):
        self._read_file()
        return self._dict_shapes, self._skipped_shaped
    
class writeXMLfile():
    def __init__(self,
                 file: str,
                 verbose: int = 3):

        self._input_floc: str = file
        self._verbose: int = verbose
        self._dict_xmlfile: dict = self._read_file()
    
    def _read_file(self):
        dict_data = defaultdict(list)
        
        with open(self._input_floc, 'r') as fin:
            bool_preamble = False
            bool_shape = False
            shape_id = None
            
            print(self._input_floc)
            for line in fin.readlines():
                line = line.strip()
                if '<Shape_' not in line and bool_preamble == False and bool_shape==False:
                    bool_preamble = True
                    
                if '<Shape_' in line:
                    bool_shape = True
                    bool_preamble =  False
                    shape_id = line.replace('<', '').replace('>', '')
                    
                if '</Shape_' in line:
                    bool_shape = False
                    dict_data[shape_id].append(f'</{shape_id}>')
                    continue
                
                if bool_shape:
                    dict_data[shape_id].append(line)
                elif bool_preamble:
                    if '</ImageData>' == line:
                        continue
                    dict_data['preamble'].append(line)
        
        dict_data['closure'].append('</ImageData>')
        
        return dict_data
    
    def write_xml_file(self, lst_shape_ids, cluster_name):
        
        with open(cluster_name+'.xml', 'w') as fout:
            lst_shapes = [y for x in lst_shape_ids for y in self._dict_xmlfile[x] if 'Shape' in x]
            lst_out = self._dict_xmlfile['preamble'] + lst_shapes + self._dict_xmlfile['closure']
            fout.writelines('\n'.join(lst_out))


def read_from_xml(uploaded_fobj, n_coordinates_min) -> dict[str, list]:
    st.toast('Reading uploaded XML file.')
    plaque_xml = parseXMLfile(uploaded_fobj, n_coordinates_min, 0)
    dict_shapes, n_shapes_skipped = plaque_xml.shapes
    st.toast(f'Finished reading XML file, skipped {n_shapes_skipped} shapes.')
    return dict_shapes
                    
def write_to_xml(df_data: 'pd.DataFrame', uploadedfile, fig):
    with tempfile.TemporaryDirectory() as tempdir:
        uploaded_floc = os.path.join(tempdir, uploadedfile.name)
        exp_name = uploadedfile.name
        exp_name = exp_name.replace('.xml', '__')

        with open(uploaded_floc, "wb") as f:
            f.write(uploadedfile.getbuffer())
            
        #! Get rid of this in the future!
        dt = str(datetime.now().strftime('%Y%m%d-%H%M'))
        with open(os.path.join('/home/asi/research/3_msomics/spatialproteomics/data/external/uploaded/', uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

        export_xml = writeXMLfile(uploaded_floc)
        zip_file_name = os.path.join(tempdir, 'results.zip')

        fig_file_name = os.path.join(
            tempdir, f'{exp_name}plaque_subregions.png')
        fig.savefig(fig_file_name, dpi=400)

        lst_files = [fig_file_name]
        
        #! XML files are empty
        for cluster_name, df_cluster in df_data.groupby('final_cl_name'):
            cluster_name = os.path.join(tempdir, f'{exp_name}{cluster_name}')
            lst_shapes = [f'Shape_{x}' for x in df_cluster.index.tolist()]

            export_xml.write_xml_file(lst_shapes, cluster_name)
            lst_files.append(cluster_name + '.xml')

        with ZipFile(zip_file_name, 'w') as zip_obj:
            for f_loc in lst_files:
                zip_obj.write(f_loc)

        dt = str(datetime.now().strftime('%Y%m%d-%H%M'))
        st.download_button(label="Download results",
                           data=open(zip_file_name, 'rb'),
                           file_name=f'{exp_name}{dt}.zip',
                           mime='application/zip')
