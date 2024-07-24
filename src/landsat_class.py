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

    def data_convert(self, band_number: str, band_dn: np.ndarray):
        """
        Convert DN data to reflectance
        Warning: this process may be wrong now
        """
        if self.dataset_level == 'L1':
            # TODO Landsat 8 stray light correction
            # ( https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files
            # /LSDS-1574_L8_Data_Users_Handbook-v5.0.pdf )

            # https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

            # (https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files
            # /LSDS-1415_Landsat4-5-TM-C2-L1-DFCB-v3.pdf)
            # (https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files
            # /LSDS-1414_Landsat7ETM-C2-L1-DFCB-v3.pdf)
            # (https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files
            # /LSDS-1822_Landsat8-9-OLI-TIRS-C2-L1-DFCB-v6.pdf)

            # convert to top of atmosphere (TOA) reflectance
            # out = (mp * DN + ap) / sin(se)
            mp = None
            ap = None
            se = None
            with open(self.input_path + f'/{self.mtl_txt}', 'r') as mtl_file:
                data = mtl_file.readlines()
            pattern_mp = r'REFLECTANCE_MULT_BAND_' + band_number + ' = (.*)E(.*)'
            pattern_ap = r'REFLECTANCE_ADD_BAND_' + band_number + ' = (.*)'
            pattern_se = r'SUN_ELEVATION = (.*)'
            for ele in data:
                mp_select = re.search(pattern_mp, ele)
                if mp_select:
                    mp = float(mp_select.group(1)) * np.pow(10, float(mp_select.group(2)))
                ap_select = re.search(pattern_ap, ele)
                if ap_select:
                    ap = float(ap_select.group(1))
                se_select = re.search(pattern_se, ele)
                if se_select:
                    se = float(se_select.group(1))
            band_array = (mp * band_dn + ap) / np.sin(se / 180)
            band_effective_region = np.logical_and(band_array >= 0, band_array <= 1)
        else:
            # dataset_level == 'L2'

            # (https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files
            # /LSDS-1618_Landsat-4-7_C2-L2-ScienceProductGuide-v4.pdf)
            # (https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files
            # /LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf)

            # convert to surface reflectance
            band_array = band_dn * 0.0000275 - 0.2
            band_effective_region = np.logical_and(band_dn >= 7273, band_dn <= 43636)

        return band_array, band_effective_region

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
                    case 'LANDSAT_5' | 'LANDSAT_7':
                        self.band_info_list = ['1', '2', '3', '4', '5', '7']
                    case 'LANDSAT_8' | 'LANDSAT_9':
                        self.band_info_list = ['1', '2', '3', '4', '5', '6', '7']
                    case _:
                        raise RuntimeError('Unknown dataset type!')
                # Finish search of dataset type
                break

        # Get dataset_collection
        dataset_collection = None
        pattern = r'COLLECTION_NUMBER = ([0-9]*)'
        for ele in data:
            dataset_collection_select = re.search(pattern, ele)
            if dataset_collection_select:
                dataset_collection = dataset_collection_select.group(1)
                # Finish search of dataset collection
                break
        if dataset_collection != '02':
            raise RuntimeError('Collection 1 data is deprecated. Consider to use Collection 2 data instead.')

        # Get dataset_level
        # PROCESSING_LEVEL = "L1TP" or "L2SP"
        pattern = r'PROCESSING_LEVEL = "L([12])[ST]P?"'
        for ele in data:
            dataset_level_select = re.search(pattern, ele)
            if dataset_level_select:
                dataset_level = dataset_level_select.group(1)
                if dataset_level == '1':
                    self.dataset_level = 'L1'
                    warnings.warn('Using Level 1 data may be not accurate. '
                                  'Consider to use Level 2 data instead.'
                                  , category=RuntimeWarning)
                else:
                    self.dataset_level = 'L2'
                # Finish search of dataset level
                break

        # Get needed tif data
        for i in self.band_info_list:
            for ele in data:
                if self.dataset_level == 'L1':
                    pattern = r'FILE_NAME_BAND_' + i + ' = "(.*_B[0-9].TIF)"'
                else:
                    # dataset_level == 'L2'
                    pattern = r'FILE_NAME_BAND_' + i + ' = "(.*SR_B[0-9].TIF)"'
                band_select = re.search(pattern, ele)
                if band_select:
                    with gdal.Open(self.input_path + f'/{band_select.group(1)}') as tiff_select:
                        if self.dataset_level == 'L1' and self.dataset_type in ('LANDSAT_5', 'LANDSAT_7'):
                            data_type = np.uint8
                        else:
                            data_type = np.uint16
                        band_dn = tiff_select.GetRasterBand(1).ReadAsArray().astype(data_type)
                        band_array, band_effective_region = self.data_convert(i, band_dn)
                        # TODO: QA_PIXEL correction
                        self.bands.append(band_array)
                        if self.effective_region is None:
                            self.effective_region = band_effective_region
                        else:
                            self.effective_region = np.logical_and(self.effective_region, band_effective_region)

                        if self.geoTransform is None:
                            self.geoTransform = tiff_select.GetGeoTransform()
                            self.projection = tiff_select.GetProjection()
                            self.tiff_size = [tiff_select.RasterXSize, tiff_select.RasterYSize]
                    # Search next band
                    break
