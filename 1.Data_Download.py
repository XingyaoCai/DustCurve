import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 全域設定 ---
FILE_PATH_STR = os.path.expanduser('~/DJAv4.2')
DJA_ROOT_URL_STR = "https://s3.amazonaws.com/msaexp-nirspec/extractions"
DJA_V4_CATALOG_PATH_STR = './DJAv4.2Catalog.csv'

def download_single_fits(session, url, full_path, dir_path):
    """
    使用 requests.Session 下載單一檔案。

    返回:
        - str: 成功下載的檔案路徑
        - None: 如果下載失敗或檔案已存在
    """
    try:
        # 檢查檔案是否已存在，避免重複下載
        if os.path.exists(full_path):
            return None # 回傳 None 表示跳過

        # 建立目錄 (如果不存在)
        # 這樣做可以避免多個執行緒同時嘗試建立同一個目錄而引發的競爭條件
        os.makedirs(dir_path, exist_ok=True)

        # 使用 session 發送 GET 請求
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status() # 如果 HTTP 狀態碼不是 200，則引發異常
            with open(full_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return full_path
    except requests.RequestException as e:
        # print(f"下載失敗: {url}, 錯誤: {e}")
        return None
    except Exception as e:
        # print(f"處理檔案時發生錯誤: {url}, 錯誤: {e}")
        return None

def main():
    # 建立主儲存目錄
    os.makedirs(FILE_PATH_STR, exist_ok=True)

    # 讀取和排序目錄
    dja_v4_catalog_df = pd.read_csv(DJA_V4_CATALOG_PATH_STR)
    dja_v4_catalog_df = dja_v4_catalog_df.sort_values(by='root')

    print(f"\n{'='*60}")
    print("開始下載資料")
    print(f"{'='*60}\n")

    total_files_int = len(dja_v4_catalog_df)
    print(f"總共需要處理的檔案數量: {total_files_int}")

    # 準備下載任務列表
    tasks = []
    for index, row in dja_v4_catalog_df.iterrows():
        object_root_str = row.root
        object_filename_str = row.file

        fits_file_root_path_str = os.path.join(FILE_PATH_STR, object_root_str)
        fits_file_full_path_str = os.path.join(fits_file_root_path_str, object_filename_str)
        fits_file_url_str = f"{DJA_ROOT_URL_STR}/{object_root_str}/{object_filename_str}"

        tasks.append((fits_file_url_str, fits_file_full_path_str, fits_file_root_path_str))

    successful_downloads = 0
    skipped_downloads = 0

    # 使用 ThreadPoolExecutor 進行並行下載
    # 對於 I/O 密集型任務，執行緒數量可以設定得比 CPU 核心數多
    # 這裡設定為 64，您可以根據您的網路狀況調整
    max_workers = 64
    print(f"使用 {max_workers} 個執行緒進行並行下載。")

    # requests.Session() 可以重複使用 TCP 連線，提高效率
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 建立 future 物件
            future_to_url = {executor.submit(download_single_fits, session, url, path, dir_path): url for url, path, dir_path in tasks}

            # 使用 tqdm 顯示進度條
            for future in tqdm(as_completed(future_to_url), total=total_files_int, desc="下載 FITS 檔案中"):
                result = future.result()
                if result is not None:
                    # 如果 result 不是 None，表示下載成功
                    successful_downloads += 1
                else:
                    # 如果是 None，表示檔案已存在或下載失敗
                    skipped_downloads +=1

    failed_downloads = total_files_int - successful_downloads - skipped_downloads

    print(f"\n{'='*60}")
    print("下載完成。")
    print(f"成功下載的新檔案: {successful_downloads}")
    print(f"已存在而跳過的檔案: {skipped_downloads}")
    print(f"下載失敗的檔案: {failed_downloads}")
    print(f"總檔案數: {total_files_int}")
    print(f"新下載成功率: {successful_downloads/total_files_int*100:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

