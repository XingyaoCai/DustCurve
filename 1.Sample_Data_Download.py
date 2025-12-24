import warnings

import astropy.units
import FunctionLib as FL
import inspect
from tqdm import tqdm
import astropy
import wave
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import re
import scipy

mpl.rcParams['font.family'] = 'serif'


warnings.filterwarnings("ignore")



import multiprocessing as mp
import os
from functools import partial
import pandas as pd
from tqdm import tqdm

# The target function for multiprocessing must be at the top level
def download_single_file(args):
    """
    Wrapper function that unpacks arguments for downloading a single file.
    This approach avoids issues with partial functions in multiprocessing.
    """
    index, catalog_row, root_url, file_path = args
    try:
        Object_Root_str = catalog_row.root
        Object_FileName_str = catalog_row.file

        # 创建完整的保存路径
        Fits_File_Root_Path_str = os.path.join(file_path, Object_Root_str)
        Fits_File_Full_Path_str = os.path.join(
            Fits_File_Root_Path_str, Object_FileName_str)
        Fits_File_Url_str = f"{root_url}/{Object_Root_str}/{Object_FileName_str}"

        # 如果目录不存在，则创建
        os.makedirs(Fits_File_Root_Path_str, exist_ok=True)

        # 检查文件是否已存在
        if os.path.exists(Fits_File_Full_Path_str):
            return 0  # 文件已存在，跳过下载

        # 使用 curl 进行下载 (macOS 兼容性更好)
        result = os.system(
            f'curl -s -o "{Fits_File_Full_Path_str}" "{Fits_File_Url_str}"')

        if result == 0:
            return 0  # 下载成功
        else:
            print(f"curl 下载失败，返回码: {result} for {Fits_File_Url_str}")
            return -1 # 下载失败

    except Exception as e:
        print(f"下载文件时出错 (索引 {index}): {e}")
        return -1 # 下载失败

def main():

    DJAv4Catalog = FL.Spectrum_Catalog()
    DJAv4Catalog.load_from_pkl(os.path.expanduser(
    './DJAV4.2Catalog.pkl'))
    print(DJAv4Catalog.sample_num())

    File_Path_str = os.path.expanduser('~/DJAv4.2Mini')
    DJA_Root_Url_str = "https://s3.amazonaws.com/msaexp-nirspec/extractions"
    os.makedirs(File_Path_str, exist_ok=True)
    DJA_v4_Catalog_Path_str = './DJAv4.2Catalog.csv'
    DJA_v4_Catalog_DataFrame = pd.read_csv(DJA_v4_Catalog_Path_str)

    download_data_list=list()
    for id, catalog in DJAv4Catalog.catalog_iterator():
        if not catalog['sample_flag']:
            continue

        prism_file_name=catalog['prism_filepath'].split('/')[-1]
        download_data_list.append((prism_file_name))

        for filter, grating_path in catalog['grating_filepaths'].items():
            grating_file_name = grating_path.split('/')[-1]
            download_data_list.append((grating_file_name))

    target_files_list=np.array(download_data_list)

    # 根据文件名列表筛选目录 DataFrame
    filtered_catalog_df = DJA_v4_Catalog_DataFrame[
        DJA_v4_Catalog_DataFrame['file'].isin(target_files_list)
    ].copy() # 使用 .copy() 避免 SettingWithCopyWarning

    # 检查是否有未在目录中找到的文件
    found_files = set(filtered_catalog_df['file'])
    not_found_files = [f for f in target_files_list if f not in found_files]

    if not_found_files:
        print(f"\n{'!'*20} 警告 {'!'*20}")
        print("以下文件未在 CSV 目录中找到，将不会被下载:")
        for f in not_found_files:
            print(f"- {f}")
        print(f"{'!'*50}\n")

    if filtered_catalog_df.empty:
        print("根据您的列表，没有找到任何可下载的文件。请检查您的文件名是否正确。")
        return
    # ======================================================

    print(f"\n{'='*60}")
    print(f"准备下载数据")
    print(f"{'='*60}\n")

    total_files_int = len(filtered_catalog_df)
    print(f"在您的列表中共找到 {total_files_int} 个文件准备下载。")

    # 为每个下载任务准备参数
    download_args = []
    for idx, row in filtered_catalog_df.iterrows():
        download_args.append((idx, row, DJA_Root_Url_str, File_Path_str))

    # 使用多进程并行下载
    num_processes_int = int(mp.cpu_count() * 0.8)
    print(f"使用 {num_processes_int} 个进程进行并行下载。")

    with mp.Pool(processes=num_processes_int) as pool:
        results = list(tqdm(pool.imap(download_single_file, download_args),
                            total=total_files_int,
                            desc="下载 FITS 文件"))

    successful_downloads = sum(1 for result in results if result == 0)
    failed_downloads = sum(1 for result in results if result == -1)

    print(f"\n{'='*60}")
    print(f"下载完成。")
    print(f"成功下载文件数: {successful_downloads}")
    print(f"下载失败文件数: {failed_downloads}")
    print(f"总计需要下载文件数: {total_files_int}")
    if total_files_int > 0:
        print(f"成功率: {successful_downloads/total_files_int*100:.2f}%")
        print(f"失败率: {failed_downloads/total_files_int*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 显式设置启动方法为 'spawn' 以确保 macOS 兼容性
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 在某些环境中启动方法可能已被设置
    main()
