import collections
import re
import tqdm
import pandas as pd
import os
import numpy as np
import astropy.units

# 假设 Load_Spectrum_Redshift 是一个已定义的函数
# from your_module import Load_Spectrum_Redshift

class Spectrum_Catalog:
    def __init__(self):
        # 初始化一个默认的字段集合
        self.default_fields = {
            'survey_id_subid': None,
            'prism_filepath': None,
            'prism_redshift': None,
            'determined_redshift': None,
            'grating_filepaths': {},
            'grating_redshifts': {},
            'file_count': 0,
            'available_filters': set(),
            'properties': {}
        }
        # 使用一个函数来创建新的条目，这个函数会用到当前的默认字段
        self.catalog = collections.defaultdict(lambda: self.default_fields.copy())

        self.filepath_pattern_re = r'^/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^_]+)_([^_]+)_([a-zA-Z0-9_]+)\.spec.fits$'

    def add_property_field(self, field_name, default_value=None):
        """
        为所有 catalog 条目动态添加一个新的顶级字段。

        Parameters
        ----------
        field_name : str
            要添加的新字段的名称。
        default_value : any, optional
            新字段的默认值, by default None
        """
        if field_name not in self.default_fields:
            # 更新默认字段模板，以便新创建的条目也包含这个字段
            self.default_fields[field_name] = default_value
            # 遍历所有现有条目，为它们添加这个新字段
            for survey_id in self.catalog:
                if field_name not in self.catalog[survey_id]:
                    self.catalog[survey_id][field_name] = default_value

    def update_entry_property(self, survey_id_subid, property_name, value):
        """
        更新指定条目的一个顶级属性。

        如果该属性不存在，它会根据需要被添加。

        Parameters
        ----------
        survey_id_subid : str
            要更新的条目的 ID。
        property_name : str
            要更新的属性的名称。
        value : any
            要设置的新值。
        """
        if survey_id_subid not in self.catalog:
            raise KeyError(f"ID '{survey_id_subid}' 在 catalog 中未找到。")

        # 检查这个字段是否是已知的字段，如果不是，则将其添加到所有条目中
        if property_name not in self.default_fields:
            self.add_property_field(property_name)

        self.catalog[survey_id_subid][property_name] = value


    def process_files(self, filepath_list, DJA_Catalog_DataFrame=None):
        """
        Process a list of file paths and populate the catalog with spectrum information.

        Parameters
        ----------
        filepath_list : list of str
            A list of file paths to process.

        DJA_Catalog_DataFrame : pd.DataFrame, optional
            A DataFrame containing the catalog data, used to load redshift information for prism spectra.
        If provided, it will be used to load the redshift for prism spectra.
        If None, the redshift will not be loaded for prism spectra.

        Returns
        -------
        None
        """

        for filepath_str in tqdm.tqdm(filepath_list):
            match = re.match(self.filepath_pattern_re, filepath_str)

            if match:
                re_group = match.groups()
                survey_name = re_group[4]
                filter_name = re_group[5]
                survey_id_subid = f"{survey_name}_{re_group[6]}"

                entry = self.catalog[survey_id_subid]
                entry['survey_id'] = survey_name
                entry['survey_id_subid'] = survey_id_subid
                entry['file_count'] += 1
                entry['available_filters'].add(filter_name)

                if filter_name == 'prism-clear':
                    entry['prism_filepath'] = filepath_str
                    # entry['prism_redshift'] = Load_Spectrum_Redshift(
                    #     filepath_str, DJA_Catalog_DataFrame)
                else:
                    entry['grating_filepaths'][filter_name] = filepath_str
                    # entry['grating_redshifts'][filter_name] = Load_Spectrum_Redshift(
                    #     filepath_str, DJA_Catalog_DataFrame)

    def load_spectrum_info(self, survey_id_subid):
        """
        Load the spectrum information for a given survey_id_subid.

        Parameters
        ----------
        survey_id_subid : str
            The survey_id_subid of the spectrum to retrieve.

        Returns
        -------
        dict
            A dictionary containing the spectrum information, or None if not found.
        """
        return dict(self.catalog.get(survey_id_subid, None))

    def load_spectrums_with_prism(self):
        """
        Load a list of survey_id_subid that have prism spectra.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding prism file paths as values.
        """
        return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is not None}

    def load_spectrums_with_grating(self):
        """
        Get a list of survey_id_subid that have grating spectra.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
        """
        return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['grating_filepaths']}

    def load_spectrums_missing_prism(self):
        """
        Get a list of survey_id_subid that do not have prism spectra.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
        """
        return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is None}

    def get_summary_stats(self):
        """
        Get summary statistics of the catalog.

        Returns
        -------
        dict
            A dictionary containing the total number of spectra, number of unique objects, and number of available filters.
        """
        all_filters = set()

        total_objects = len(self.catalog)
        with_prism = len(self.load_spectrums_with_prism())
        with_grating = len(self.load_spectrums_with_grating())
        without_prism = len(self.load_spectrums_missing_prism())
        total_spectra = sum(entry['file_count']
                            for entry in self.catalog.values())
        total_grating_spectra = sum(
            len(entry['grating_filepaths']) for entry in self.catalog.values())
        total_prism_spectra = sum(
            1 for entry in self.catalog.values() if entry['prism_filepath'] is not None)

        for entry in self.catalog.values():
            all_filters.update(entry['available_filters'])

        return {
            'total_objects': total_objects,
            'with_prism': with_prism,
            'without_prism': without_prism,
            'with_grating': with_grating,
            'total_spectra': total_spectra,
            'total_prism_spectra': total_prism_spectra,
            'total_grating_spectra': total_grating_spectra
        }

    def to_dataframe(self):
        """
        Convert the catalog to a pandas DataFrame with dictionaries preserved.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the spectrum information with dictionaries and sets preserved.
        """
        # data = []

        # for survey_id_subid, entry in self.catalog.items():
        #     row = {
        #         'survey_id_subid': survey_id_subid,
        #         'survey_id': entry['survey_id'],
        #         'prism_filepath': entry['prism_filepath'],
        #         'prism_redshift': entry['prism_redshift'],
        #         # Keep as dict
        #         'grating_filepaths': entry['grating_filepaths'],
        #         # Keep as dict
        #         'grating_redshifts': entry['grating_redshifts'],
        #         'determined_redshift': entry['determined_redshift'],
        #         'file_count': entry['file_count'],
        #         'available_filters': entry['available_filters'],  # Keep as set
        #         'properties': entry['properties']  # Keep as dict
        #     }
        #     data.append(row)

        if not self.catalog:
            return pd.DataFrame()

        # 将 catalog 字典转换为 DataFrame
        df = pd.DataFrame.from_dict(self.catalog, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'survey_id_subid'}, inplace=True)
        return df


    def save_catalog_to_pkl(self, filename):
        """
        Save the catalog to a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to save the catalog to.
        """
        df = self.to_dataframe()
        df.to_pickle(filename)

    def load_from_pkl(self, pkl_filepath):
        """
        Load catalog data from a pickle file.

        Parameters
        ----------
        pkl_filepath : str
            Path to the pickle file to load.

        Returns
        -------
        None
        """
        if not os.path.exists(pkl_filepath):
            raise FileNotFoundError(f"The file {pkl_filepath} does not exist.")
        if not pkl_filepath.endswith('.pkl'):
            raise ValueError(
                f"The file {pkl_filepath} is not a valid pickle file.")

        df = pd.read_pickle(pkl_filepath)

        # 从 DataFrame 的列中推断出所有的字段
        self.default_fields = {col: None for col in df.columns if col != 'survey_id_subid'}
        # 确保一些字段是正确的类型
        self.default_fields['grating_filepaths'] = {}
        self.default_fields['grating_redshifts'] = {}
        self.default_fields['available_filters'] = set()
        self.default_fields['properties'] = {}

        self.catalog.clear()

        # 将 DataFrame 转换为 catalog 字典
        for _, row in df.iterrows():
            survey_id_subid = row['survey_id_subid']
            # 使用 to_dict() 来获取行数据，然后转换为 defaultdict 所期望的普通 dict
            self.catalog[survey_id_subid] = dict(row)

    def find_complete_objects(self, required_filters=None):
        """
        Find objects that have all required filters and a prism file.

        Parameters
        ----------
        required_filters : list of str, optional
            A list of filter names that must be present for an object to be considered complete.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding catalog entries as values.
        """
        if required_filters is None:
            required_filters = set()
        else:
            required_filters = set(required_filters)

        complete_objects = {}
        for survey_id_subid, entry in self.catalog.items():
            if (entry['prism_filepath'] is not None and required_filters.issubset(entry['available_filters'])):
                complete_objects[survey_id_subid] = entry

        return complete_objects

    def catalog_iterator(self):
        """
        Returns an iterator over the catalog entries.

        Yields
        -------
        tuple
            A tuple containing the survey_id_subid and the corresponding catalog entry.
        """
        for index in self.catalog.keys():
            yield index, self.catalog[index]

    def determine_redshift(self, survey_id_subid):
        """
        Determine the redshift for a given survey_id_subid. And fill the `determined_redshift` field in the catalog entry.
        If two redshifts are available and within 5% of each other, it will use the average of the redshifts. If the difference is greater than 5%, it will use the prism redshift if available, or the grating redshift if only one is available.
        If more than one grating redshift is available, it will choose the average of the two most similar redshifts.
        If no redshift is available, it will return None.
        If only one redshift is available, it will use that one.
        Parameters
        ----------
        survey_id_subid : str
            The survey_id_subid of the spectrum to determine the redshift for.
        Returns
        -------
        astropy.units.Quantity or None
            The determined redshift as an astropy.units.Quantity object, or None if no redshift could be determined.
        """
        if survey_id_subid not in self.catalog:
            raise ValueError(
                f"Survey ID {survey_id_subid} not found in the catalog.")

        entry = self.catalog[survey_id_subid]
        entry['properties']['redshift_conflict'] = False

        prism_redshift = entry['prism_redshift']
        grating_redshifts = [
            value for value in entry['grating_redshifts'].values() if not np.isnan(value)]

        # No redshifts available
        if prism_redshift is None and not grating_redshifts:
            return None

        # Only prism redshift available
        if prism_redshift is not None and not grating_redshifts:
            entry['determined_redshift'] = prism_redshift
            return prism_redshift

        # Only one grating redshift available, no prism
        if prism_redshift is None and len(grating_redshifts) == 1:
            entry['determined_redshift'] = grating_redshifts[0]
            return grating_redshifts[0]

        # One prism and one grating redshift
        if prism_redshift is not None and len(grating_redshifts) == 1:
            if abs(prism_redshift - grating_redshifts[0]) / prism_redshift < 0.05:
                entry['determined_redshift'] = (
                    prism_redshift + grating_redshifts[0]) / 2
                return entry['determined_redshift']
            else:
                entry['determined_redshift'] = prism_redshift
                entry['properties']['redshift_conflict'] = True
                return prism_redshift

        # Multiple redshifts available (either prism + multiple grating, or just multiple grating)
        all_redshifts = []
        if prism_redshift is not None:
            all_redshifts.append(prism_redshift)
        all_redshifts.extend(grating_redshifts)

        # Find the two most similar redshifts
        min_diff = float('inf')
        best_pair = None
        for i in range(len(all_redshifts)):
            for j in range(i + 1, len(all_redshifts)):
                diff = abs(all_redshifts[i] - all_redshifts[j])
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (all_redshifts[i], all_redshifts[j])

        if best_pair:
            # Check if the two nearest redshifts are within 5%
            relative_diff = abs(
                best_pair[0] - best_pair[1]) / min(best_pair[0], best_pair[1])
            avg_redshift = (best_pair[0] + best_pair[1]) / 2

            if relative_diff >= 0.05:
                entry['properties']['redshift_conflict'] = True

            entry['determined_redshift'] = avg_redshift
            return avg_redshift
        else:
            # Fallback (should not happen if we have redshifts)
            entry['determined_redshift'] = all_redshifts[0]
            return all_redshifts[0]

    def update_catalog_item(self, id, catalog):
        """
        Update a catalog item with the given id and catalog data.

        Parameters
        ----------
        id : str
            The survey_id_subid of the spectrum to update.
        catalog : dict
            The catalog data to update the item with.
        """
        if id not in self.catalog:
            raise ValueError(f"Survey ID {id} not found in the catalog.")

        self.catalog[id].update(catalog)

    def sample_num(self):
        """
        Returns the number of samples in the catalog.

        Returns
        -------
        int
            The number of samples in the catalog.
        """
        count = 0
        for id, catalog in self.catalog.items():
            if catalog.get('Sample_Flag') is True: # 使用 .get() 更安全
                count += 1
        return count


    def __repr__(self):
        """
        Returns a panda DataFrame representation of the catalog.
        """
        df = self.to_dataframe()
        if df.empty:
            return "Spectrum_Catalog is empty."
        else:
            return df.to_string(index=False, max_rows=10, max_colwidth=50, justify='left') + "\n\n" + f"Total objects: {len(self.catalog)}"
