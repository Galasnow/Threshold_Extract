import argparse
import gc
from multiprocessing import Pool
from functools import partial

import cv2 as cv
import matplotlib.pyplot as plt
from osgeo import gdal

from landsat_class import Landsat
from all_function import *

# import pycuda
# conda install pycuda

draw_image_flag = 1
output_flag = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config.yml')
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    '''
        In water index calculation, 
        gets the best threshold and full area of water,
        according to known parts or approximation of water.
    '''

    args = parse_args()
    config_file = args.config
    input_path, mtl_txt, index, evaluation_method, threshold_range, final_precision, cores, beta_sq \
        = read_config(config_file)
    landsat_obj = Landsat(input_path, mtl_txt)
    landsat_obj.read_information()

    # Calculate water Index
    index_range, index_array = water_index(landsat_obj.bands, index, landsat_obj.dataset_type,
                                           landsat_obj.effective_region)
    print('index.min() = ', index_range[0])
    print('index.max() = ', index_range[1])

    # Write the index result to tif
    if output_flag:
        # reference: https://github.com/waveletswave/NDWI-WaterAnalysis
        # Get the GDAL driver for GeoTIFF format
        driver = gdal.GetDriverByName('GTiff')

        # Create a new GeoTIFF file to store the result
        with driver.Create(input_path + f'/{index}' + '.tif', landsat_obj.tiff_size[0], landsat_obj.tiff_size[1],
                           1, gdal.GDT_Float32) as out_tiff:
            # Set the geotransform and projection information for the out TIFF based on the input tif
            out_tiff.SetGeoTransform(landsat_obj.geoTransform)
            out_tiff.SetProjection(landsat_obj.projection)

            # Write the out array to the first band of the new TIFF
            out_tiff.GetRasterBand(1).WriteArray(index_array)

            # Write the data to disk
            out_tiff.FlushCache()

    _, mydwi_array = water_index(landsat_obj.bands, 'MyDWI', landsat_obj.dataset_type,
                                 landsat_obj.effective_region)
    # Set the true value of water. Temporarily use manual setting instead.
    match (landsat_obj.dataset_type, landsat_obj.dataset_level):
        # Test case, not accurate
        case('LANDSAT_5', 'L1'):
            value = binarization(mydwi_array, 0.11)
        case('LANDSAT_5', 'L2'):
            value = binarization(mydwi_array, -0.24)
        case ('LANDSAT_7', 'L1'):
            value = binarization(mydwi_array, 0.06)
        case ('LANDSAT_7', 'L2'):
            value = binarization(mydwi_array, -0.23)
        case ('LANDSAT_8', 'L1'):
            value = binarization(mydwi_array, 0.42)
        case ('LANDSAT_8', 'L2'):
            value = binarization(mydwi_array, -0.25)
        case ('LANDSAT_9', 'L1'):
            value = binarization(mydwi_array, 0.36)
        case ('LANDSAT_9', 'L2'):
            value = binarization(mydwi_array, -0.13)

    # Manually gc to free memory
    del landsat_obj.bands, mydwi_array
    gc.collect()

    if draw_image_flag:
        # Histogram
        index_1d = index_array[landsat_obj.effective_region]
        plt.hist(index_1d, bins=500, range=index_range, histtype='step')
        plt.title('Histogram')
        plt.xlabel('value')
        plt.ylabel('count')
        plt.show()

    # Create mask to mark interested area.
    mask = np.zeros_like(index_array, dtype=np.bool_)
    # Set value of interested area to 1. Temporarily use manual setting instead.
    match landsat_obj.dataset_type:
        # Test case
        case 'LANDSAT_5':
            mask[100:1600, 1400:2700] = np.ones((1500, 1300), dtype=np.bool_)
            mask[2000:3000, 2700:3700] = np.ones((1000, 1000), dtype=np.bool_)
            mask[2800:3800, 5100:5900] = np.ones((1000, 800), dtype=np.bool_)
            mask[4300:5700, 5500:6500] = np.ones((1400, 1000), dtype=np.bool_)
        case 'LANDSAT_7':
            mask[100:1600, 1400:2700] = np.ones((1500, 1300), dtype=np.bool_)
            mask[2000:3000, 2700:3700] = np.ones((1000, 1000), dtype=np.bool_)
            mask[2500:4000, 5500:6500] = np.ones((1500, 1000), dtype=np.bool_)
            mask[2800:4000, 4800:5800] = np.ones((1200, 1000), dtype=np.bool_)
        case 'LANDSAT_8':
            mask[200:2000, 1300:2500] = np.ones((1800, 1200), dtype=np.bool_)
            mask[2000:3400, 2400:3200] = np.ones((1400, 800), dtype=np.bool_)
            mask[3000:4200, 4800:5600] = np.ones((1200, 800), dtype=np.bool_)
            mask[4800:7000, 2500:3800] = np.ones((2200, 1300), dtype=np.bool_)
        case 'LANDSAT_9':
            mask[300:1800, 1300:2500] = np.ones((1500, 1200), dtype=np.bool_)
            mask[2000:3400, 2400:3200] = np.ones((1400, 800), dtype=np.bool_)
            mask[3000:4200, 4800:5700] = np.ones((1200, 900), dtype=np.bool_)
            mask[4800:7000, 2500:3800] = np.ones((2200, 1300), dtype=np.bool_)

    mask = np.logical_and(mask, landsat_obj.effective_region)

    time_start = time.time()

    # Define the search range of threshold. Manually set the range may speed up and get more accurate result
    if threshold_range is None:
        threshold_range = index_range
    else:
        threshold_range = range_intersect(threshold_range, index_range)

    # Define the number of process in multiprocessing. This number should change according to final precision,
    # decreasing step_size means heavier task, and then increasing cores may speed up the total task.
    # Normally, suitable number may be 2 to 6
    if cores is None:
        cores = 2

    # Define the final precision and step_size list
    primary_step_size = (threshold_range[1] - threshold_range[0]) / (cores * 2)
    # The final precision of threshold
    if final_precision is None:
        final_precision = primary_step_size / 100

    multiple = primary_step_size / final_precision
    iterations = int(np.log10(multiple)) + 1
    # [primary_step_size, final_step_size]
    step_size_l = np.logspace(np.log10(primary_step_size), np.log10(final_precision), iterations)
    print('step_size_l = ', step_size_l)

    # Iteration. In every iteration, set [best_threshold - half_interval, best_threshold + half_interval] as new range,
    # and reduce step_size, increase half_threshold, to get accurate result faster.
    best_threshold = None
    half_interval = None
    range_expand_ratio = 2
    result_list_all = []
    for step_size in step_size_l:
        print()
        if best_threshold is None or half_interval is None:
            threshold_range_iter = threshold_range
        else:
            # best_threshold is last best_threshold
            # half_interval is last step_size * 2
            threshold_range_iter = range_intersect(
                [best_threshold - half_interval, best_threshold + half_interval],
                index_range)
        print('threshold_range = ', threshold_range_iter)
        print('step_size = ', step_size)

        range_list = range_divide(threshold_range_iter, cores)
        # MultiProcessing
        with Pool(processes=cores) as pool:
            func = partial(saliency_evaluation, index_array, value, mask, step_size, evaluation_method,
                           beta_sq=beta_sq)
            print('Processing...')
            result: list = pool.map(func, range_list)
        # Remove a layer of bracket
        s: list = sum(result, [])

        result_list = np.array(s)
        print('result_list = ', result_list)
        # Extract the best threshold, max measure value
        max_measure = np.max(result_list[:, 3])
        print('max', evaluation_method, ' = ', max_measure)
        max_index = np.argmax(result_list[:, 3])
        best_threshold = result_list[max_index, 0]
        print('best_threshold = ', best_threshold)

        result_list_all.extend(result_list)
        # Expand the range every iteration to avoid falling into local optimal solution
        half_interval = step_size * range_expand_ratio
        range_expand_ratio *= 1.5

    result_list_all = np.array(result_list_all)
    # Sort the result according to first column (threshold)
    result_index = np.argsort(result_list_all[:, 0])
    result_list_all = result_list_all[result_index, :]
    # print('result_list_all = ', result_list_all)

    # Binarization according to the best threshold
    index_array_bin = binarization(index_array, best_threshold)

    time_end = time.time()
    print('threshold extraction time is : ' + str(time_end - time_start) + ' s')

    if draw_image_flag:
        # Create window and show image
        show = index_array_bin.astype(np.float32)

        # # Draw mask box
        # top = 4800
        # bottom = 7000
        # left = 2500
        # right = 3800
        # half_width = 3
        # show = draw_box(show, top, bottom, left, right, half_width, 1)

        if output_flag:
            show[show == 1] = 255
            cv.imwrite(input_path + f'/{index}' + '.jpg', show, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        cv.namedWindow('binary_image', cv.WINDOW_NORMAL)
        cv.imshow('binary_image', show)
        # Press 'X' to close and continue
        while cv.getWindowProperty('binary_image', cv.WND_PROP_VISIBLE) > 0:
            # show image repeatedly for 100ms
            cv.waitKey(100)
        else:
            # Release window
            cv.destroyAllWindows()

    if draw_image_flag:
        # Measure-Threshold curve
        threshold_l = result_list_all[:, 0]
        measure_l = result_list_all[:, 3]

        plt.plot(threshold_l, measure_l, 'bo-', alpha=0.5, linewidth=1, label=index)
        plt.legend()  # show label
        plt.title('Measure-Threshold curve')
        plt.xlabel('threshold')  # x_label
        plt.ylabel(evaluation_method)  # y_label
        plt.show()

        # P-R curve
        precision_l = result_list_all[:, 1]
        recall_l = result_list_all[:, 2]

        plt.plot(recall_l, precision_l, 'bo-', alpha=0.5, linewidth=1, label=index)
        plt.legend()  # show label
        plt.title('P-R curve')
        plt.xlabel('Recall')  # x_label
        plt.ylabel('Precision')  # y_label
        plt.xlim(0, 1)  # set range of axis x
        plt.ylim(0, 1)  # set range of axis y
        plt.show()

    # Compare the result with global OTSU threshold
    # OTSU ####################################################################
    time_start = time.time()

    threshold_range = index_range

    # Define the final precision and step_size list
    primary_step_size = (threshold_range[1] - threshold_range[0]) / (cores * 2)
    # The final precision of threshold
    if final_precision is None:
        final_precision = primary_step_size / 100

    multiple = primary_step_size / final_precision
    iterations = int(np.log10(multiple)) + 1
    # [primary_step_size, final_step_size]
    step_size_l = np.logspace(np.log10(primary_step_size), np.log10(final_precision), iterations)
    print('\n', '#####OTSU#####')
    print('step_size_l = ', step_size_l)

    # Iteration. In every iteration, set [best_threshold - half_interval, best_threshold + half_interval] as new range,
    # and reduce step_size, increase half_threshold, to get accurate result faster.
    best_threshold = None
    half_interval = None
    range_expand_ratio = 2
    result_list_all = []
    for step_size in step_size_l:
        print()
        if best_threshold is None or half_interval is None:
            threshold_range_iter = threshold_range
        else:
            # best_threshold is last best_threshold
            # half_interval is last step_size * 2
            threshold_range_iter = range_intersect(
                [best_threshold - half_interval, best_threshold + half_interval],
                index_range)
        print('threshold_range = ', threshold_range_iter)
        print('step_size = ', step_size)

        range_list = range_divide(threshold_range_iter, cores)
        # MultiProcessing
        with Pool(processes=cores) as pool:
            func = partial(OTSU, index_array, landsat_obj.effective_region, step_size)
            print('Processing...')
            result = pool.map(func, range_list)
        # Remove a layer of bracket
        s: list = sum(result, [])

        result_list = np.array(s)
        print('result_list = ', result_list)

        # Extract the best threshold, max var
        max_var = np.max(result_list[:, 1])
        print('max var = ', max_var)
        max_Index = np.argmax(result_list[:, 1])
        best_threshold = result_list[max_Index, 0]
        print('best_threshold = ', best_threshold)

        result_list_all.extend(result_list)
        # Expand the range every iteration to avoid falling into local optimal solution
        half_interval = step_size * range_expand_ratio
        range_expand_ratio *= 1.5

    result_list_all = np.array(result_list_all)
    # Sort the result according to first column (threshold)
    result_index = np.argsort(result_list_all[:, 0])
    result_list_all = result_list_all[result_index, :]
    # print('result_list_all = ', result_list_all)

    time_end = time.time()
    print('OTSU time is : ' + str(time_end - time_start) + ' s')

    # Binarization
    otsu_img = binarization(index_array, best_threshold) * 255

    if output_flag:
        otsu_img.astype(np.uint8)
        cv.imwrite(input_path + f'/{index}' + '_OTSU.jpg', otsu_img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    if draw_image_flag:
        # Create window and show image
        show = otsu_img.astype(np.float32)
        cv.namedWindow('OTSU_image', cv.WINDOW_NORMAL)
        cv.imshow('OTSU_image', show)
        # Press 'X' to close and continue
        while cv.getWindowProperty('OTSU_image', cv.WND_PROP_VISIBLE) > 0:
            # Show image repeatedly for 100ms
            cv.waitKey(100)
        else:
            # Release window
            cv.destroyAllWindows()

    if draw_image_flag:
        # Var-Threshold curve
        threshold_l = result_list_all[:, 0]
        var_l = result_list_all[:, 1]

        plt.plot(threshold_l, var_l, 'bo-', alpha=0.5, linewidth=1, label=index)
        plt.legend()  # show label
        plt.title('Var-Threshold curve')
        plt.xlabel('threshold')  # x_label
        plt.ylabel('var')  # y_label
        plt.show()

    # otsu_threshold, otsu_image = cv.threshold(image_original, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
