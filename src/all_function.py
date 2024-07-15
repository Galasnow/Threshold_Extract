import time
from typing import Callable

import yaml
import numpy as np
from typing import Sequence

runtime_data_type = np.float16
debug_print_flag = 0


def time_it(func: Callable):
    # decorator used to calculate time, use @time_it in front of any function definition
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = func(*args, **kwargs)
        time_end = time.time()
        print('consumed time of "', getattr(func, "__name__"), '" is : ', str(time_end - time_start) + ' s')
        return ret

    return wrapper


def read_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    path = config['path']
    mtl_txt = config['mtl_txt']
    index = config['index']
    evaluation_method = config['evaluation_method']
    threshold_range = config['threshold_range']
    final_precision = config['final_precision']
    cores = config['cores']
    beta_sq = config['F-Measure']['beta_sq']
    return path, mtl_txt, index, evaluation_method, threshold_range, final_precision, cores, beta_sq


def water_index(bands: Sequence[np.ndarray], index: str, dataset_type: str, effective_region: np.ndarray[np.bool_]) \
        -> (list, np.ndarray):
    """ Calculate Water index """
    match index:
        case 'NDWI':
            # https://doi.org/10.1080/01431169608948714
            # Calculate Normalized Difference Water Index (NDWI)
            # MNDWI = (Green - NIR) / (Green + NIR)
            match dataset_type:
                case 'LANDSAT_5' | 'Landsat7':
                    numerate = bands[1] - bands[3]
                    denominator = bands[1] + bands[3]
                case 'LANDSAT_8' | 'LANDSAT_9':
                    numerate = bands[2] - bands[4]
                    denominator = bands[2] + bands[4]
                case _:
                    print('unknown dataset type!')
                    exit(-1)

            result = np.divide(numerate, denominator, out=np.full_like(denominator, np.inf, dtype=runtime_data_type),
                               where=effective_region)

        case 'MNDWI':
            # https://doi.org/10.1080/01431160600589179
            # Calculate Modified Normalized Difference Water Index (MNDWI)
            # MNDWI = (Green - MIR) / (Green + MIR)
            match dataset_type:
                case 'LANDSAT_5' | 'Landsat7':
                    numerate = bands[1] - bands[4]
                    denominator = bands[1] + bands[4]
                case 'LANDSAT_8' | 'LANDSAT_9':
                    numerate = bands[2] - bands[5]
                    denominator = bands[2] + bands[5]
                case _:
                    print('unknown dataset type!')
                    exit(-1)

            result = np.divide(numerate, denominator, out=np.full_like(denominator, np.inf, dtype=runtime_data_type),
                               where=effective_region)

        case 'AWEI-nsh':
            # https://doi.org/10.1016/j.rse.2013.08.029.
            # Calculate Automated Water Extraction Index (AWEI) (no shadow version)
            match dataset_type:
                case 'LANDSAT_5' | 'Landsat7':
                    index_value = 4 * bands[1] - 0.25 * bands[3] - 4 * bands[4] + 2.75 * bands[5]
                    result = np.where(effective_region, index_value, np.inf)
                case 'LANDSAT_8' | 'LANDSAT_9':
                    index_value = 4 * bands[2] - 0.25 * bands[4] - 4 * bands[5] + 2.75 * bands[6]
                    result = np.where(effective_region, index_value, np.inf)
                case _:
                    print('unknown dataset type!')
                    exit(-1)

        case 'AWEI-sh':
            # Calculate Automated Water Extraction Index (AWEI) (shadow version)
            match dataset_type:
                case 'LANDSAT_5' | 'Landsat7':
                    index_value = bands[0] + 2.5 * bands[1] - 1.5 * bands[3] - 1.5 * bands[4] - 0.25 * bands[5]
                    result = np.where(effective_region, index_value, np.inf)
                case 'LANDSAT_8' | 'LANDSAT_9':
                    index_value = bands[1] + 2.5 * bands[2] - 1.5 * bands[4] - 1.5 * bands[5] - 0.25 * bands[6]
                    result = np.where(effective_region, index_value, np.inf)
                case _:
                    print('unknown dataset type!')
                    exit(-1)

        case 'MyDWI':
            # Calculate My Difference Water Index (MyDWI)
            match dataset_type:
                case 'LANDSAT_5' | 'Landsat7':
                    numerate = bands[0] + bands[1] - bands[3] - bands[4] - bands[5]
                    denominator = bands[0] + bands[1] + bands[3] + bands[4] + bands[5]
                case 'LANDSAT_8' | 'LANDSAT_9':
                    numerate = bands[1] + bands[2] - bands[4] - bands[5] - bands[6]
                    denominator = bands[1] + bands[2] + bands[4] + bands[5] + bands[6]
                case _:
                    print('unknown dataset type!')
                    exit(-1)

            result = np.divide(numerate, denominator, out=np.full_like(denominator, np.inf, dtype=runtime_data_type),
                               where=effective_region)
        case _:
            print('unknown index!')
            exit(-1)

    # record minimum first,
    # avoid following search of minimum under the interference of background(-(abs(result_min)*100))
    result_min = result.min()
    background_value = -(abs(result_min) * 100)
    result = np.nan_to_num(result, posinf=background_value)
    result_max = result.max()
    return [result_min, result_max], result


def binarization(image: np.ndarray, threshold, positive=True) -> np.ndarray:
    if positive:
        result = (image > threshold).astype(np.bool_)
    else:
        result = (image < threshold).astype(np.bool_)
    return result


def range_divide(threshold_range: Sequence, number: int) -> list:
    bottom = threshold_range[0]
    top = threshold_range[1]
    length = (top - bottom) / number
    range_list = []
    for i in np.linspace(bottom, top, number, endpoint=False):
        range_list.append([i, i + length])
    if debug_print_flag:
        print('range_list = ', range_list)
    return range_list


def saliency_evaluation(image: np.ndarray, value: np.ndarray, mask: np.ndarray, step_size, method: str,
                        threshold_range: Sequence,
                        beta_sq=0.3):
    result_list = []
    for threshold in np.arange(threshold_range[0], threshold_range[1], step_size):

        binary_image = binarization(image, threshold)

        # Ues logical operation seems faster and consumes less memory than multiply and bitwise_and
        mask_image = np.logical_and(binary_image, mask)
        mask_value = np.logical_and(value, mask)
        intersect_result = np.logical_and(mask_image, mask_value)

        # Sum the elements equal to 'True'
        num0 = np.sum(intersect_result)  # TP
        num1 = np.sum(mask_image)  # TP+FP
        num2 = np.sum(mask_value)  # TP+FN

        if debug_print_flag:
            print('num0 = ', num0)
            print('num1 = ', num1)
            print('num2 = ', num2)

        tp = num0
        fp = num1 - num0
        fn = num2 - num0

        precision = np.divide(tp, tp + fp, where=((tp + fp) != 0))
        recall = np.divide(tp, tp + fn, where=((tp + fn) != 0))

        if debug_print_flag:
            print('Precision = ', precision)
            print('Recall = ', recall)

        match method:
            case 'F-Measure':
                # https://doi.org/10.1016/j.rse.2013.08.029
                # beta_sq = beta ** 2
                numerate = (1 + beta_sq) * precision * recall
                denominator = beta_sq * precision + recall
                measure = np.divide(numerate, denominator, where=(denominator != 0))

            case 'E-Measure':  # correct?
                # https://doi.org/10.24963/ijcai.2018/97
                # https://github.com/DengPingFan/E-measure
                element_count = np.sum(mask)
                if num2 == element_count:
                    enhanced_matrix = 1.0 - mask_image
                elif num2 == 0:
                    enhanced_matrix = mask_image
                else:
                    mean_image = num1 / element_count
                    mean_value = num2 / element_count
                    align_image = np.subtract(mask_image, mean_image)
                    align_value = np.subtract(mask_value, mean_value)
                    numerate = np.multiply(2 * align_image, align_value)
                    denominator = (np.multiply(align_image, align_image) + np.multiply(align_value, align_value) +
                                   np.finfo(runtime_data_type).eps)
                    align_matrix = np.divide(numerate, denominator)
                    enhanced_matrix = np.divide(np.multiply(align_matrix + 1, align_matrix + 1), 4)

                measure = np.sum(enhanced_matrix) / (element_count - 1 + np.finfo(runtime_data_type).eps)

            # TODO
            # 'IOU', 'GIOU', 'CIOU', '  CM', 'FBw', 'VQ', 'S-Measure' ...
            case _:
                print('Unsupported evaluation method!')
                exit(-1)

        if debug_print_flag:
            print('threshold = ', threshold)
            print(method, ' = ', measure)

        result_list.append([threshold, precision, recall, measure])
    if debug_print_flag:
        print('result_list = ', result_list)

    return result_list


def OTSU(gray_img: np.ndarray, effective_region: np.ndarray[np.bool_], step_size, threshold_range: Sequence):
    # https://blog.csdn.net/qq_15534667/article/details/112551471

    threshold_list = np.arange(threshold_range[0], threshold_range[1], step_size)

    num = len(threshold_list)
    gray_level0 = np.zeros(num, dtype=np.float32)
    gray_level1 = np.zeros(num, dtype=np.float32)
    w0 = np.zeros(num, dtype=np.float32)

    effective_count = np.sum(effective_region)
    #
    i = 0
    for threshold in threshold_list:
        # mean of foreground
        gray_level0_position = np.logical_and((gray_img > threshold), effective_region)
        # foreground/(foreground+background)
        w0[i] = np.shape(gray_img[gray_level0_position])[0] / effective_count

        # mean of background
        gray_level1_position = np.logical_and((gray_img <= threshold), effective_region)

        gray_level0[i] = np.mean(gray_img[gray_level0_position])
        gray_level1[i] = np.mean(gray_img[gray_level1_position])
        i += 1

    if debug_print_flag:
        print('gray_level0 =', gray_level0)
        print('gray_level1 =', gray_level1)

    w1 = 1 - w0
    # var
    var = np.array(w0 * w1 * (gray_level0 - gray_level1) ** 2).astype(np.float32)
    result = []

    for i in range(0, num):
        result.append([threshold_list[i], var[i]])

    if debug_print_flag:
        print(result)

    return result


def draw_box(image: np.ndarray, top: int, bottom: int, left: int, right: int, half_width: int, value) -> np.ndarray:
    size = image.shape
    if ((bottom + half_width > size[0]) or (right + half_width > size[1])
            or (left - half_width < 0) or (top - half_width < 0)):
        print('exceed boundary!')
        return image
    image[top: bottom, :][:, left - half_width:left + half_width] = value
    image[top: bottom, :][:, right - half_width:right + half_width] = value
    image[top - half_width:top + half_width, :][:, left:right] = value
    image[bottom - half_width:bottom + half_width, :][:, left: right] = value
    return image
