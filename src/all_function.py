import time
from typing import Callable
import warnings

import yaml
import numpy as np
from typing import Sequence

runtime_data_type = np.float32
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
            # NDWI = (Green - NIR) / (Green + NIR)
            match dataset_type:
                case 'LANDSAT_5' | 'LANDSAT_7':
                    numerate = bands[1] - bands[3]
                    denominator = bands[1] + bands[3]
                case 'LANDSAT_8' | 'LANDSAT_9':
                    numerate = bands[2] - bands[4]
                    denominator = bands[2] + bands[4]
                case _:
                    raise RuntimeError('Unknown dataset type!')

            result = np.divide(numerate, denominator, out=np.full_like(denominator, np.inf, dtype=runtime_data_type),
                               where=effective_region)

        case 'MNDWI':
            # https://doi.org/10.1080/01431160600589179
            # Calculate Modified Normalized Difference Water Index (MNDWI)
            # MNDWI = (Green - MIR) / (Green + MIR)
            match dataset_type:
                case 'LANDSAT_5' | 'LANDSAT_7':
                    numerate = bands[1] - bands[4]
                    denominator = bands[1] + bands[4]
                case 'LANDSAT_8' | 'LANDSAT_9':
                    numerate = bands[2] - bands[5]
                    denominator = bands[2] + bands[5]
                case _:
                    raise RuntimeError('Unknown dataset type!')

            result = np.divide(numerate, denominator, out=np.full_like(denominator, np.inf, dtype=runtime_data_type),
                               where=effective_region)

        case 'AWEI-nsh':
            # https://doi.org/10.1016/j.rse.2013.08.029
            # Calculate Automated Water Extraction Index (AWEI) (no shadow version)
            match dataset_type:
                case 'LANDSAT_5' | 'LANDSAT_7':
                    index_value = 4 * bands[1] - 0.25 * bands[3] - 4 * bands[4] + 2.75 * bands[5]
                    result = np.where(effective_region, index_value, np.inf)
                case 'LANDSAT_8' | 'LANDSAT_9':
                    index_value = 4 * bands[2] - 0.25 * bands[4] - 4 * bands[5] + 2.75 * bands[6]
                    result = np.where(effective_region, index_value, np.inf)
                case _:
                    raise RuntimeError('Unknown dataset type!')

        case 'AWEI-sh':
            # Calculate Automated Water Extraction Index (AWEI) (shadow version)
            match dataset_type:
                case 'LANDSAT_5' | 'LANDSAT_7':
                    index_value = bands[0] + 2.5 * bands[1] - 1.5 * bands[3] - 1.5 * bands[4] - 0.25 * bands[5]
                    result = np.where(effective_region, index_value, np.inf)
                case 'LANDSAT_8' | 'LANDSAT_9':
                    index_value = bands[1] + 2.5 * bands[2] - 1.5 * bands[4] - 1.5 * bands[5] - 0.25 * bands[6]
                    result = np.where(effective_region, index_value, np.inf)
                case _:
                    raise RuntimeError('Unknown dataset type!')

        case 'WI2015':
            # http://dx.doi.org/10.1016/j.rse.2015.12.055
            # Calculate 2015 water index (WI2015)
            match dataset_type:
                case 'LANDSAT_5' | 'LANDSAT_7':
                    index_value = 1.7204 + 171 * bands[1] + 3 * bands[2] - 70 * bands[3] - 45 * bands[4] - 71 * bands[5]
                    result = np.where(effective_region, index_value, np.inf)
                case 'LANDSAT_8' | 'LANDSAT_9':
                    index_value = 1.7204 + 171 * bands[2] + 3 * bands[3] - 70 * bands[4] - 45 * bands[5] - 71 * bands[6]
                    result = np.where(effective_region, index_value, np.inf)
                case _:
                    raise RuntimeError('Unknown dataset type!')

        case 'MyDWI':
            # Calculate My Difference Water Index (MyDWI)
            match dataset_type:
                case 'LANDSAT_5' | 'LANDSAT_7':
                    numerate = bands[0] + bands[1] - bands[3] - bands[4] - bands[5]
                    denominator = bands[0] + bands[1] + bands[3] + bands[4] + bands[5]
                case 'LANDSAT_8' | 'LANDSAT_9':
                    numerate = bands[1] + bands[2] - bands[4] - bands[5] - bands[6]
                    denominator = bands[1] + bands[2] + bands[4] + bands[5] + bands[6]
                case _:
                    raise RuntimeError('unknown dataset type!')

            result = np.divide(numerate, denominator, out=np.full_like(denominator, np.inf, dtype=runtime_data_type),
                               where=effective_region)
        case _:
            raise NotImplementedError('Unsupported index!')

    # Record minimum first,
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
    range_list = [[i, i+length] for i in np.linspace(bottom, top, number, endpoint=False)]
    if debug_print_flag:
        print('range_list = ', range_list)
    return range_list


def range_intersect(range_1: Sequence, range_2: Sequence):
    """
        Return intersection of 2 ranges
        Examples:
            range_1 = [1, 3]
            range_2 = [0, 2]
            return [1, 2]
    """
    out_range_min = max(range_1[0], range_2[0])
    out_range_max = min(range_1[1], range_2[1])
    if out_range_min > out_range_max:
        raise RuntimeError('Given threshold range is completely out of range of water index!')
    return [out_range_min, out_range_max]


def saliency_evaluation(image: np.ndarray, value: np.ndarray, mask: np.ndarray, step_size, method: str,
                        threshold_range: Sequence,
                        beta_sq=0.3):
    """
    image: array of index
    value: ground truth
    mask: region of interest (ROI)
    """
    # Move operation of "value" outside for loop to speed up
    # Intersection
    mask_value = np.logical_and(value, mask)
    # Sum the elements equal to 'True'
    tp_plus_fn = np.sum(mask_value)  # TP+FN

    result_list = []
    for threshold in np.arange(threshold_range[0], threshold_range[1], step_size):

        binary_image = binarization(image, threshold)

        # Intersection. Logical operation seems faster and consumes less memory than multiply and bitwise_and
        mask_image = np.logical_and(binary_image, mask)
        intersect_result = np.logical_and(mask_image, mask_value)

        # Sum the elements equal to 'True'
        tp = np.sum(intersect_result)  # TP
        tp_plus_fp = np.sum(mask_image)  # TP+FP

        if debug_print_flag:
            print('TP = ', tp)
            print('TP + FP = ', tp_plus_fp)
            print('TP + FN = ', tp_plus_fn)

        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp

        precision = np.divide(tp, tp + fp, where=((tp + fp) != 0))
        recall = np.divide(tp, tp + fn, where=((tp + fn) != 0))

        match method:
            case 'F-Measure':
                # Cornelius J. Van Rijsbergen. 1979. Information Retrieval. Butterworth and Co., London.
                # beta_sq = beta ** 2
                numerate = (1 + beta_sq) * precision * recall
                denominator = beta_sq * precision + recall
                measure = np.divide(numerate, denominator, where=(denominator != 0))

            case 'E-Measure':  # correct?
                # https://doi.org/10.24963/ijcai.2018/97
                # https://github.com/DengPingFan/E-measure
                element_count = np.sum(mask)
                if tp_plus_fn == element_count:
                    enhanced_matrix = True - mask_image
                elif tp_plus_fn == 0:
                    enhanced_matrix = mask_image
                else:
                    mean_image = tp_plus_fp / element_count
                    mean_value = tp_plus_fn / element_count
                    align_image = mask_image - mean_image
                    align_value = mask_value - mean_value
                    numerate = 2 * align_image * align_value
                    denominator = (align_image * align_image + align_value * align_value +
                                   np.finfo(runtime_data_type).eps)
                    align_matrix = numerate / denominator
                    enhanced_matrix = (align_matrix + 1) * (align_matrix + 1) / 4

                measure = np.sum(enhanced_matrix) / (element_count - 1 + np.finfo(runtime_data_type).eps)

            case 'Kappa':
                # https://doi.org/10.1177%2F001316446002000104
                element_count = np.sum(mask)
                tn = element_count - tp - fn - fp
                p0 = (tp + tn) / element_count
                pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (element_count * element_count)
                measure = (p0 - pe) / (1 - pe)

            # TODO
            # 'CM', 'FBw', 'VQ', 'S-Measure' ...
            case _:
                raise NotImplementedError('Unsupported evaluation method!')

        if debug_print_flag:
            print('threshold = ', threshold)
            print('Precision = ', precision)
            print('Recall = ', recall)
            print(method, ' = ', measure)

        result_list.append([threshold, precision, recall, measure])
    if debug_print_flag:
        print('result_list = ', result_list)

    return result_list


def OTSU(gray_img: np.ndarray, effective_region: np.ndarray[np.bool_], step_size, threshold_range: Sequence):
    # https://blog.csdn.net/qq_15534667/article/details/112551471

    threshold_list = np.arange(threshold_range[0], threshold_range[1], step_size)

    num = threshold_list.size
    gray_level0 = np.zeros(num, dtype=np.float32)
    gray_level1 = np.zeros(num, dtype=np.float32)
    w0 = np.zeros(num, dtype=np.float32)

    effective_count = np.sum(effective_region)
    #
    i = 0
    for threshold in threshold_list:
        # locate gray level 0/1 position
        gray_level0_position = (gray_img > threshold)
        gray_level1_position = np.logical_xor(gray_level0_position, effective_region)
        # foreground / (foreground + background)
        w0[i] = np.sum(gray_level0_position) / effective_count
        # Mean of foreground and background
        gray_level0[i] = np.mean(gray_img[gray_level0_position]) if w0[i] != 0 else 0
        gray_level1[i] = np.mean(gray_img[gray_level1_position]) if w0[i] != 1 else 0
        i += 1

    # background / (foreground + background)
    w1 = 1 - w0
    # Var
    var = np.array(w0 * w1 * (gray_level0 - gray_level1) ** 2).astype(np.float32)
    var = np.nan_to_num(var, nan=0)

    result = [[threshold_list[i], var[i]] for i in range(0, num)]

    if debug_print_flag:
        print('w0 =', w0)
        print('w1 =', w1)
        print('gray_level0 =', gray_level0)
        print('gray_level1 =', gray_level1)
        print('result =', result)

    return result


def draw_box(image: np.ndarray, top: int, bottom: int, left: int, right: int, half_width: int, value) -> np.ndarray:
    size = image.shape
    if ((bottom + half_width > size[0]) or (right + half_width > size[1])
            or (left - half_width < 0) or (top - half_width < 0)):
        warnings.warn('exceed boundary!', category=RuntimeWarning)
        return image
    image[top: bottom, :][:, left - half_width:left + half_width] = value
    image[top: bottom, :][:, right - half_width:right + half_width] = value
    image[top - half_width:top + half_width, :][:, left:right] = value
    image[bottom - half_width:bottom + half_width, :][:, left: right] = value
    return image
