import re
import warnings
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


class Landsat:
    """ Landsat Class """
    # Restrict attributes
    __slots__ = ('input_path', 'mtl_txt', 'dataset_type', 'dataset_level', 'tiff_size',
                 'geoTransform', 'projection', 'band_info_list', 'bands', 'effective_region')

    def __init__(self, _input_path: str, _mtl_txt: str):
        self.input_path = _input_path
        self.mtl_txt = _mtl_txt
        self.dataset_type = None
        self.dataset_level = None
        self.tiff_size = None
        self.geoTransform = None
        self.projection = None
        self.band_info_list = None
        self.bands = []
        self.effective_region = None

    def read_information(self):
        with open(self.input_path + f'/{self.mtl_txt}', 'r') as mtl_file:
            data = mtl_file.readlines()

        # Get dataset_type
        pattern = r'SPACECRAFT_ID = "(.*)"'
        for ele in data:
            dataset_type_select = re.search(pattern, ele)
            if dataset_type_select:
                self.dataset_type = dataset_type_select.group(1)
                match self.dataset_type:
                    case 'LANDSAT_5' | 'Landsat7' | 'LANDSAT_7':
                        self.band_info_list = ['1', '2', '3', '4', '5', '7']
                    case 'LANDSAT_8' | 'LANDSAT_9':
                        self.band_info_list = ['1', '2', '3', '4', '5', '6', '7']
                    case _:
                        raise RuntimeError('Unknown dataset type!')
                # Finish search of dataset type
                break

        # Get dataset_level
        # PRODUCT_TYPE or PROCESSING_LEVEL = "L1T" or "L2SP"
        pattern = r'.*[EL] = "L([12])[ST]"'
        for ele in data:
            dataset_level_select = re.search(pattern, ele)
            if dataset_level_select:
                print(dataset_level_select)
                dataset_level = dataset_level_select.group(1)
                match dataset_level:
                    case '1':
                        self.dataset_level = 1
                        warnings.warn('Use Level 1 data may be not accurate. Consider to use Level 2 data instead.',
                                      category=RuntimeWarning)
                    case '2':
                        self.dataset_level = 2
                    case _:
                        raise RuntimeError('Unknown dataset level!')
                # Finish search of dataset level
                break
        print(self.dataset_level)

        # Get needed tif data
        for i in self.band_info_list:
            for ele in data:
                match self.dataset_type:
                    case 'LANDSAT_5' | 'LANDSAT_7' | 'LANDSAT_8' | 'LANDSAT_9':
                        pattern = r'FILE_NAME_BAND_' + i + ' = "(.*)"'
                    case 'Landsat7':
                        pattern = r'BAND' + i + '_FILE_NAME = "(.*)"'
                band_select = re.search(pattern, ele)
                if band_select:
                    with gdal.Open(self.input_path + f'/{band_select.group(1)}') as tiff_select:
                        band_select_array = tiff_select.GetRasterBand(1).ReadAsArray().astype(np.float32)
                        self.bands.append(band_select_array)
                        if self.geoTransform is None:
                            self.geoTransform = tiff_select.GetGeoTransform()
                            self.projection = tiff_select.GetProjection()
                            self.tiff_size = [tiff_select.RasterXSize, tiff_select.RasterYSize]
                        band_effective_region = np.array(band_select_array != 0)
                        if self.effective_region is None:
                            self.effective_region = band_effective_region
                        else:
                            self.effective_region = np.logical_and(self.effective_region, band_effective_region)
                    # Search next band
                    break
