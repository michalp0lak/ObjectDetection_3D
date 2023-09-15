# %%
import numpy as np
import os
import riegl.rdb
import time
import json

class RDB_import():

    def __init__(self, filepath, attributes, condition, chunk_size, only_xyz = False):

        assert (type(filepath) == str) and os.path.exists(filepath), 'RDBX file does not exist'
        assert (type(attributes) == list) and len(attributes) > 0, 'List of attributes has to be list of strings'
        assert (type(condition) == str), 'Condition has to be string value'
        assert (type(chunk_size) == int), 'Chunk_size has to be integer value'
        assert (type(only_xyz) == bool), 'Parameter only_xyz is boolean value'

        self.path = filepath
        self.attributes = attributes
        self.condition = condition
        self.chunk_size = chunk_size
        self.only_xyz = only_xyz

    def get_features(self):

        if self.only_xyz:

            with riegl.rdb.rdb_open(self.path) as rdb:

                xyz = []

                for points in rdb.select(
                    selection=self.condition,
                    attributes=self.attributes,
                    chunk_size = self.chunk_size # optional: number of points to load in one step
                ):
                
                    xyz.append(points["riegl.xyz"])
                gtag = rdb.meta_data["riegl.geo_tag"]

            rdb.close()

            # Join chunks
            mat = np.array([], dtype=np.float16).reshape(0,3)
            for chunk in xyz: mat = np.concatenate((mat, np.asarray(chunk, dtype=np.float64)), axis=0)

            return mat, gtag

        else:

            with riegl.rdb.rdb_open(self.path) as rdb:
                
                # Because XYZ values are triplets and other attributes univariate, they need to be collected separetly for concatenation
                # This would be probably possible to solve in more elegant way
                xyz_frame = []

                attribute_frame = []
                for points in rdb.select(
                    selection=self.condition,
                    attributes=self.attributes,
                    chunk_size = self.chunk_size  # optional: number of points to load in one step
                ):
                
                    xyz_frame.append(points["riegl.xyz"])
                    
                    attr_collector = []
                    
                    for attr in self.attributes[1:]: attr_collector.append(points[attr])
                   
                    attribute_frame.append(attr_collector)
                gtag = rdb.meta_data["riegl.geo_tag"]

            rdb.close()

            # Join chunks
            xyz_mat = np.array([], dtype=np.float16).reshape(0,3)
            attr_mat = np.array([], dtype=np.float16).reshape(0,len(self.attributes)-1)

            for chunk in xyz_frame: xyz_mat = np.concatenate((xyz_mat, np.asarray(chunk, dtype=np.float64)), axis=0)
            for chunk in attribute_frame: attr_mat = np.concatenate((attr_mat, np.asarray(chunk, dtype=np.float64).T), axis=0)

            # Concatenate XYZ and other attributes
            mat = np.concatenate((xyz_mat, attr_mat), axis=1)
   
            return mat, gtag


class RDB_export():

    def __init__(self, filepath, attributes, attributes_dtype, attributes_matrix, chunk_size, geo_tag):


        assert (type(filepath) == str), 'Path of RDBX featurized file does not exist'
        assert (type(attributes) == dict) and len(attributes) == 2, 'Parameter attributes has to be dictionary with 2 elements'
        assert ((type(attributes['BuiltIn']) == list) and (type(attributes['Custom']) == list)) and (len(attributes['BuiltIn'])) > 0, 'Both elements of attributes dictionary are lists and there has to be at least one (xyz) attribute in a list of BuiltIn attributes'
        assert (type(attributes_dtype) == dict) and len(attributes_dtype) == 2, 'Attribute dtypes has to be dictionary with 2 elements'
        assert ((type(attributes_dtype['BuiltIn']) == dict) and (type(attributes_dtype['Custom']) == dict)) and (len(attributes_dtype['BuiltIn'])) > 0, 'Both elements of attribute_dtypes dictionary are dictionaries and there has to be at least one (xyz) attribute dtype in a dict of BuiltIn attributes dtypes'
        assert (type(attributes_matrix) == np.ndarray) and (len(attributes_matrix.shape) == 2), 'Matrix of attributes has to be 2-dimensional numpy array'
        assert (type(chunk_size) == int), 'Chunk_size has to be integer value'

        self.path = filepath
        self.attributes = attributes
        self.attributes_dtype = attributes_dtype
        self.mat = attributes_matrix
        self.chunk_size = chunk_size
        self.gtag = geo_tag

    def chunk(self, mat, n):

        # loop over the matrix in n-sized chunks
        for i in range(0, mat.shape[0], n):
            # yield the current n-sized chunk to the calling function
            yield mat[i: i + n,:]

    def define_custom_attributes(self, pointcloud):
    
        """
        This function shows how to define and add new point attributes
        and is used by the example functions below. Please note that there
        is also a shortcut for built-in RIEGL default point attributes
        which we use to define the "riegl.class" attribute at the end of
        this function.
        """

        # Point built-in attributes - by using a shortcut for built-in RIEGL attributes:
        #pointcloud.point_attributes.add("riegl.id")

        for builtIn_attr in self.attributes['BuiltIn'][1:]: pointcloud.point_attributes.add(builtIn_attr)


        for custom_attr in self.attributes['Custom']:
            #print("{} created".format(custom_attr))
            custom_var = custom_attr
            custom_var = riegl.rdb.PointAttribute(pointcloud)
            #custom_var.name = "riegl.{}".format(custom_attr)
            custom_var.name = custom_attr
            custom_var.title = custom_attr
            custom_var.description = "Feature: {}".format(custom_attr)
            custom_var.unit_symbol = ""
            custom_var.length = 1
            custom_var.resolution = 0.001
            custom_var.minimum_value = -1e5  # minimum,
            custom_var.maximum_value = 1e5  # maximum and
            custom_var.default_value = 0.000  # default
            custom_var.storage_class = riegl.rdb.PointAttribute.StorageClass.VARIABLE
            pointcloud.point_attributes.add(custom_var)

    def create_RDB(self):

        """
        Create RDB file with custom features
        """

        # New RDB library context
        context = riegl.rdb.Context()

        # New database instance
        rdb = riegl.rdb.Pointcloud(context)

        # Create new point cloud database
        settings = riegl.rdb.CreateSettings(context)

        # Define primary point attribute, usually the point coordinates
        # details see class riegl::rdb::pointcloud::PointAttribute
        settings.primary_attribute.name = "riegl.xyz"
        settings.primary_attribute.title = "XYZ"
        settings.primary_attribute.description = "Cartesian point coordinates"
        settings.primary_attribute.unit_symbol = "m"
        settings.primary_attribute.length = 3
        settings.primary_attribute.resolution = 0.00025
        settings.primary_attribute.minimum_value = -535000.0  # minimum,
        settings.primary_attribute.maximum_value = +535000.0  # maximum and
        settings.primary_attribute.default_value = 0.0  # default in m
        settings.primary_attribute.storage_class = \
            riegl.rdb.PointAttribute.StorageClass.VARIABLE

        # Define database settings
        settings.chunk_size = 50000  # maximum number of points per chunk
        settings.compression_level = 50  # 50% compression rate

        # Finally create new database
        rdb.create(self.path, settings)


        attr_list = self.attributes['BuiltIn'] + self.attributes['Custom']
        data_types = self.attributes_dtype['BuiltIn'] | self.attributes_dtype['Custom']

        # Before we can modify the database, we must start a transaction
        with riegl.rdb.Transaction(
            rdb,  # point cloud to create transaction for
            "Initialization",  # transaction title
            "Custom RDBX point cloud featurization"  # software name
        ) as transaction:

            # Define some custom point attributes
            self.define_custom_attributes(rdb)

            #Start inserting points
            with rdb.insert() as insert:

                for chunk in self.chunk(self.mat, self.chunk_size):

                    # Create buffers for all point attributes
                    buffers = riegl.rdb.PointBuffer(rdb, count=chunk.shape[0], attributes=attr_list)

                    # It is always expected that attributes matrix columns are ordered in a way:
                    # (x, y, z, BuiltIn attributes, Custom attributes)

                    # So first three columns are copied into riegl.xyz attribute
                    # Copy xyz data to point attribute buffers
                    np.copyto(buffers["riegl.xyz"].data, chunk[:,:3])

                    # All other attributes are exported in RDBX file as univariate attributes in for loop (one-by-one)
                    for i, attr in enumerate(attr_list[1:]): 
                        #print('Attribute {} exported'.format(attr))
                        np.copyto(buffers[attr].data, chunk[:,i+3].astype(data_types[attr]))

                    # Actually insert points
                    insert.bind(buffers)
                    insert.next(chunk.shape[0])

                rdb.meta_data.set("riegl.geo_tag", self.gtag)

                # Finally commit transaction
                transaction.commit()


class BBX_HANDLER():

    def __init__(self, filepath):

        assert (type(filepath) == str), 'BBX file does not exist'

        self.path = filepath

    def bbx_json_convert(self):

        with riegl.rdb.rdb_open(self.path) as rdb:

            bbx = []

            for points in rdb.select(
                "",
                ["riegl.xyz","riegl.id", 'riegl.bbx_length_a', 'riegl.bbx_length_b', 'riegl.bbx_length_c', 
                "riegl.bbx_angle_a", "riegl.bbx_angle_b", "riegl.bbx_angle_c"],
                chunk_size = 1
            ): bbx.append([points["riegl.id"], points["riegl.xyz"], points["riegl.bbx_length_a"],
                        points["riegl.bbx_length_b"], points["riegl.bbx_length_c"], points["riegl.bbx_angle_a"],
                        points["riegl.bbx_angle_b"], points["riegl.bbx_angle_c"]])
        rdb.close()

        bbx_list = []
        for bb in bbx: bbx_list.append({'id': np.asarray(bb[0]).squeeze().tolist(), 'center': np.asarray(bb[1]).squeeze().tolist(), 
                                        'length_x': np.asarray(bb[2]).squeeze().tolist(), 
                                        'length_y': np.asarray(bb[3]).squeeze().tolist(), 
                                        'length_z': np.asarray(bb[4]).squeeze().tolist(),
                                        'angle_x': np.asarray(bb[5]).squeeze().tolist(), 
                                        'angle_y': np.asarray(bb[6]).squeeze().tolist(), 
                                        'angle_z': np.asarray(bb[7]).squeeze().tolist(),
                                    })

        with open('{}.json'.format(self.path.split('.')[0]), mode='w', encoding='utf-8') as f: json.dump(bbx_list, f)

    def collect_bbx_data(self):

        with riegl.rdb.rdb_open(self.path) as rdb:

            bbx = []

            for points in rdb.select(
                "",
                ["riegl.xyz","riegl.id", 'riegl.bbx_length_a', 'riegl.bbx_length_b', 'riegl.bbx_length_c', 
                "riegl.bbx_angle_a", "riegl.bbx_angle_b", "riegl.bbx_angle_c"],
                chunk_size = 1
            ): bbx.append([points["riegl.id"], points["riegl.xyz"], points["riegl.bbx_length_a"],
                        points["riegl.bbx_length_b"], points["riegl.bbx_length_c"], points["riegl.bbx_angle_a"],
                        points["riegl.bbx_angle_b"], points["riegl.bbx_angle_c"]])

        bbx_list = []
        for bb in bbx: bbx_list.append({'id': np.asarray(bb[0]).squeeze().tolist(), 'center': np.asarray(bb[1]).squeeze().tolist(), 
                                        'length_x': np.asarray(bb[2]).squeeze().tolist(), 
                                        'length_y': np.asarray(bb[3]).squeeze().tolist(), 
                                        'length_z': np.asarray(bb[4]).squeeze().tolist(),
                                        'angle_x': np.asarray(bb[5]).squeeze().tolist(), 
                                        'angle_y': np.asarray(bb[6]).squeeze().tolist(), 
                                        'angle_z': np.asarray(bb[7]).squeeze().tolist(),
                                    })


        return bbx_list


    def create_BBX(self, geo_tag, bbxs):

        """
        Create BBX file 
        """

        #print('EXPORT OF BBX')

        # New RDB library context
        context = riegl.rdb.Context()

        # New database instance
        rdb = riegl.rdb.Pointcloud(context)

        # Create new point cloud database
        settings = riegl.rdb.CreateSettings(context)

        # Define primary point attribute, usually the point coordinates
        # details see class riegl::rdb::pointcloud::PointAttribute
        settings.primary_attribute.name = "riegl.xyz"
        settings.primary_attribute.title = "XYZ"
        settings.primary_attribute.description = "Cartesian point coordinates"
        settings.primary_attribute.unit_symbol = "m"
        settings.primary_attribute.length = 3
        settings.primary_attribute.resolution = 0.00025
        settings.primary_attribute.minimum_value = -535000.0  # minimum,
        settings.primary_attribute.maximum_value = +535000.0  # maximum and
        settings.primary_attribute.default_value = 0.0  # default in m
        settings.primary_attribute.storage_class = \
            riegl.rdb.PointAttribute.StorageClass.VARIABLE

        # Define database settings
        settings.chunk_size = 100000  # maximum number of points per chunk
        settings.compression_level = 10  # 50% compression rate

        # Finally create new database
        rdb.create(self.path, settings)


        attr_list = ["riegl.bbx_angle_a", "riegl.bbx_angle_b", "riegl.bbx_angle_c", 
                     "riegl.bbx_length_a", "riegl.bbx_length_b", "riegl.bbx_length_c"]

        attr_limits = {"riegl.bbx_angle_a": (0,360), "riegl.bbx_angle_b": (0,360), "riegl.bbx_angle_c": (0.0,360), 
                       "riegl.bbx_length_a": (0,1000), "riegl.bbx_length_b": (0,1000), "riegl.bbx_length_c": (0,1000)} 

        attr_units = {"riegl.bbx_angle_a": 'deg', "riegl.bbx_angle_b": 'deg', "riegl.bbx_angle_c": 'deg', 
                     "riegl.bbx_length_a": 'm', "riegl.bbx_length_b": 'm', "riegl.bbx_length_c": 'm'}

        attr_description = {"riegl.bbx_angle_a": 'angle of rotation around axis a', 
                            "riegl.bbx_angle_b": 'angle of rotation around axis b', 
                            "riegl.bbx_angle_c": 'angle of rotation around axis c', 
                            "riegl.bbx_length_a": 'Length of axis a', 
                            "riegl.bbx_length_b": 'Length of axis b', 
                            "riegl.bbx_length_c": 'Length of axis c'}

        attr_resolution =  {"riegl.bbx_angle_a": 0.0001, 
                            "riegl.bbx_angle_b": 0.0001, 
                            "riegl.bbx_angle_c": 0.0001, 
                            "riegl.bbx_length_a": 0.001, 
                            "riegl.bbx_length_b": 0.001, 
                            "riegl.bbx_length_c": 0.001}
        
        # Before we can modify the database, we must start a transaction
        with riegl.rdb.Transaction(
            rdb,  # point cloud to create transaction for
            "Initialization",  # transaction title
            "Bounding boxes of trunks in point cloud"  # software name
        ) as transaction:

            # Define some bbx attributes
            rdb.point_attributes.add("riegl.selected")
            rdb.point_attributes.add("riegl.visible")

            for attr in attr_list:
                
                var = attr
                var = riegl.rdb.PointAttribute(rdb)
                var.name = attr
                var.title = attr
                var.description = attr_description[attr]
                var.unit_symbol = attr_units[attr]
                var.length = 1
                var.resolution = attr_resolution[attr]
                var.minimum_value = attr_limits[attr][0]
                var.maximum_value = attr_limits[attr][1] 
                var.default_value = 0.000  # default
                var.storage_class = riegl.rdb.PointAttribute.StorageClass.VARIABLE
                rdb.point_attributes.add(var)

            #Start inserting points
            with rdb.insert() as insert:

                for chunk_raw in bbxs:
                   
                    chunk = np.array([
                    chunk_raw['center'][0],
                    chunk_raw['center'][1],
                    chunk_raw['center'][2],
                    chunk_raw['angle_x'],
                    chunk_raw['angle_y'],
                    chunk_raw['angle_z'],
                    chunk_raw['length_x'],
                    chunk_raw['length_y'],
                    chunk_raw['length_z'],
                    ])
   
                    # Create buffers for all point attributes
                    buffers = riegl.rdb.PointBuffer(rdb, count=1, attributes=["riegl.xyz"]+attr_list)

                    # It is always expected that attributes of bbx is vector are ordered in a way:
                    # (center_x, center_y, center_z, angle_x, angle_y, angle_z, leght_x, leght_y, , leght_z)

                    # So first three values are copied into riegl.xyz attribute
                    # Copy xyz data to point attribute buffers
                    np.copyto(buffers["riegl.xyz"].data, chunk[:3].reshape(1,-1))

                    # All other attributes are exported in RDBX file as univariate attributes in for loop (one-by-one)
                    for i, attr in enumerate(attr_list): 
                        #print('Attribute {} exported'.format(attr))
                        np.copyto(buffers[attr].data, chunk[i+3].astype(np.float32))

                    # Actually insert points
                    insert.bind(buffers)
                    insert.next(1)

                rdb.meta_data.set("riegl.geo_tag", geo_tag)

                # Finally commit transaction
                transaction.commit()


    def insert_BBX(self, bbxs, add_filename):

        attr_list = ["riegl.bbx_angle_a", "riegl.bbx_angle_b", "riegl.bbx_angle_c", 
                "riegl.bbx_length_a", "riegl.bbx_length_b", "riegl.bbx_length_c"]

         # Access existing database
        with riegl.rdb.rdb_open(self.path) as rdb:

            # Before we can modify the database, we must start a transaction
            with riegl.rdb.Transaction(
                rdb,  # point cloud object
                "Insert of BBX from file: {}".format(add_filename),  # transaction title
                "BBX_HANDLER.insert_BBX"  # software name
            ) as transaction:

                # Start inserting points
                with rdb.insert() as insert:

                    for chunk_raw in bbxs:

                        chunk = np.array([
                        chunk_raw['center'][0],
                        chunk_raw['center'][1],
                        chunk_raw['center'][2],
                        chunk_raw['angle_x'],
                        chunk_raw['angle_y'],
                        chunk_raw['angle_z'],
                        chunk_raw['length_x'],
                        chunk_raw['length_y'],
                        chunk_raw['length_z'],
                        ])


                        # Create buffers for all point attributes
                        buffers = riegl.rdb.PointBuffer(rdb, count=1, attributes=["riegl.xyz"]+attr_list)

                        # It is always expected that attributes of bbx is vector are ordered in a way:
                        # (center_x, center_y, center_z, angle_x, angle_y, angle_z, leght_x, leght_y, , leght_z)

                        # So first three values are copied into riegl.xyz attribute
                        # Copy xyz data to point attribute buffers
                        np.copyto(buffers["riegl.xyz"].data, chunk[:3].reshape(1,-1))
                        #print('Attribute {}: {} exported'.format("riegl.xyz",chunk[:3].reshape(1,-1)))
                        # All other attributes are exported in RDBX file as univariate attributes in for loop (one-by-one)
                        for i, attr in enumerate(attr_list): 
                            #print('Attribute {}: {} exported'.format(attr,chunk[i+3].reshape(-1,1)))
                            np.copyto(buffers[attr].data, chunk[i+3].reshape(-1,1).astype(np.float64))

                        # Actually insert points
                        insert.bind(buffers)
                        insert.next(1)

                    # Finally commit transaction
                    transaction.commit()