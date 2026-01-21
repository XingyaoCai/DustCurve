
class DustAttenuationAnalyst:
    def __init__(self, catalog_obj, wavelength_grid=(1350, 8000), grid_step=5.0):
        """
        初始化分析器。

        Parameters
        ----------
        catalog_obj : Spectrum_Catalog
            包含元数据的 Catalog 对象。
        wavelength_grid : tuple
            (min_wave, max_wave) 单位 Å。
        grid_step : float
            网格步长。
        """
        self.catalog = catalog_obj
        self.wavelength_grid = wavelength_grid
        self.grid_step = grid_step

        # 巴耳末递减分组条件
        self.balmer_conditions = [
            (-0.05, 0.02), (0.02, 0.10), (0.10, 0.20),
            (0.20, 0.30), (0.30, 0.45), (0.45, 0.60), (0.60, 1.00)
        ]

        # 发射线 Mask 区域
        self.mask_regions = [
            (3675, 4150), (4300, 4400), (4800, 5100),
            (5850, 5900), (6200, 6325), (6500, 6775),
            (7000, 7150), (7300, 7500)
        ]

        # 数据存储容器
        self.groups = defaultdict(list)
        self.group_mean_decrements = {}
        self.averagers = {}
        self.reference_smooth_flux = None

        # 计算结果存储
        self.attenuation_curves = {}     # 每个组的 Q_lambda
        self.effective_curve = None      # 加权平均后的 Q_lambda
        self.effective_curve_wave = None
        self.curve_mode = None           # 记录当前使用的是 smoothed 还是 raw

        # 拟合结果存储
        self.poly_coeffs = None          # 多项式系数 (针对 1/lambda)
        self.scaling_factor = None       # 归一化因子 S
        self.derived_Rv = None           # 推导出的 Rv
        self.intercept_val = 0.0         # 截距 (默认为0)

    # =========================================================================
    # 1. 数据加载与处理
    # =========================================================================

    def _calculate_balmer_decrement(self, halpha, hbeta):
        if hbeta <= 0 or halpha <= 0:
            return None
        return np.log(halpha / hbeta / 2.86)

    def _get_group_index(self, value):
        for i, (low, high) in enumerate(self.balmer_conditions):
            if low <= value < high:
                return i
        return -1

    def load_and_group_objects(self, target_ids):
        """第一步：加载 ID 并按巴耳末递减分组"""
        print(f"Categorizing {len(target_ids)} objects...")
        self.groups.clear()
        self.group_mean_decrements.clear()

        group_values = defaultdict(list)

        for survey_id in target_ids:
            entry = self.catalog.load_spectrum_info(survey_id)
            if not entry: continue

            try:
                params = entry.get('dust_curve_parameters', {})
                line_params = params.get('Best_Balmer_Decrement_Line_Parameters', {})

                ha = line_params.get('halpha_flux', 0)
                hb = line_params.get('hbeta_flux', 0)

                bd_val = self._calculate_balmer_decrement(ha, hb)

                if bd_val is not None:
                    g_idx = self._get_group_index(bd_val)
                    if g_idx != -1:
                        self.groups[g_idx].append(survey_id)
                        group_values[g_idx].append(bd_val)
            except Exception:
                continue

        for idx in sorted(self.groups.keys()):
            vals = group_values[idx]
            mean_val = np.mean(vals)
            if idx == 0: mean_val = 0.0 # 强制参考组为 0
            self.group_mean_decrements[idx] = mean_val
            print(f"Group {idx}: {len(vals)} objects, Mean BD={mean_val:.3f}")

    def compute_average_spectra(self):
        """第二步：堆叠光谱"""
        print("\nComputing average spectra...")
        self.averagers = {}

        for g_idx in sorted(self.groups.keys()):
            ids = self.groups[g_idx]
            if not ids: continue

            averager = FL.SpectrumAverager()
            success = averager.load_spectrum_catalog(ids, self.catalog, FL.Load_Spectrum_From_Fits)

            if not success:
                print(f"  Group {g_idx}: Failed to load spectra.")
                continue

            averager.create_common_wavelength_grid(self.wavelength_grid, self.grid_step)
            averager.interpolate_spectra(interpolation_method='linear')
            averager.spectrum_normalization_within_range((5400, 5600), target_flux=1.e-19)

            # 使用中值堆叠 + 百分位滤波
            averager.compute_average_spectrum(
                method='median', ignore_nan=True, use_percentile_filter=True,
                lower_percentile=16, upper_percentile=84
            )

            # 应用发射线 Mask
            averager.apply_emission_line_mask(self.mask_regions)
            self.averagers[g_idx] = averager
            print(f"  Group {g_idx}: Processed.")

    def compute_smooth_reference(self, window_len=101, poly_order=3):
        """第三步：平滑参考谱 (Group 0)"""
        if 0 not in self.averagers:
            raise ValueError("Group 0 (Reference) is missing.")

        ref_avg = self.averagers[0]
        # 使用 masked 数据作为起点
        flux_data = ref_avg.average_flux_masked.copy()

        # 线性插值填补 Mask 掉的 NaN，以便进行平滑滤波
        series = pd.Series(flux_data)
        flux_interp = series.interpolate(method='linear', limit_direction='both').to_numpy()

        self.reference_smooth_flux = savgol_filter(flux_interp, window_length=window_len, polyorder=poly_order)
        print("Reference spectrum smoothed.")

    # =========================================================================
    # 2. 核心计算：Attenuation Curves (Q_lambda)
    # =========================================================================

    # def compute_attenuation_curves(self, mode='smoothed'):
    #     """
    #     第四步：计算有效衰减曲线 Q_lambda

    #     Parameters
    #     ----------
    #     mode : str
    #         'smoothed' : 使用平滑后的 Group 0 作为参考 (推荐，减少噪声)。
    #         'raw'      : 使用原始堆叠的 Group 0 作为参考。
    #     """
    #     if mode == 'smoothed':
    #         if self.reference_smooth_flux is None:
    #             self.compute_smooth_reference()
    #         ref_flux_used = self.reference_smooth_flux
    #     elif mode == 'raw':
    #         if 0 not in self.averagers:
    #              raise ValueError("Group 0 not processed.")
    #         ref_flux_used = self.averagers[0].average_flux_masked
    #     else:
    #         raise ValueError("mode must be 'smoothed' or 'raw'")

    #     self.curve_mode = mode
    #     ref_tau = self.group_mean_decrements[0]
    #     self.attenuation_curves = {}

    #     # 计算每个组的 Q 曲线
    #     for g_idx, averager in self.averagers.items():
    #         if g_idx == 0: continue

    #         target_flux = averager.average_flux_masked
    #         target_tau = self.group_mean_decrements[g_idx]
    #         delta_tau = target_tau - ref_tau

    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             ratio = target_flux / ref_flux_used
    #             ratio[ratio <= 0] = np.nan
    #             q_curve = -np.log(ratio) / delta_tau

    #         self.attenuation_curves[g_idx] = {
    #             'wavelength': averager.common_wavelength,
    #             'Q_lambda': q_curve,
    #             'delta_tau': delta_tau
    #         }

    #     # 计算加权平均 (Effective Curve)
    #     if self.attenuation_curves:
    #         q_list = [d['Q_lambda'] for d in self.attenuation_curves.values()]
    #         w_list = [d['delta_tau'] for d in self.attenuation_curves.values()]

    #         # 关键修复：确保是数值数组而非字典列表
    #         stack_q = np.ma.masked_invalid(np.array(q_list))
    #         self.effective_curve = np.ma.average(stack_q, axis=0, weights=w_list).filled(np.nan)
    #         self.effective_curve_wave = list(self.attenuation_curves.values())[0]['wavelength']
    #         print(f"Computed effective curve using {mode} reference.")
    #     else:
    #         print("No groups available to compute curves.")

    def compute_attenuation_curves(self, mode='smoothed'):
        """第四步：计算有效衰减曲线及其误差"""
        # ... (前部分代码保持不变: 确定 ref_flux_used) ...
        if mode == 'smoothed':
            if self.reference_smooth_flux is None: self.compute_smooth_reference()
            ref_flux_used = self.reference_smooth_flux
        elif mode == 'raw':
            if 0 not in self.averagers: raise ValueError("Group 0 missing.")
            ref_flux_used = self.averagers[0].average_flux_masked
        else:
            raise ValueError("mode error")

        self.curve_mode = mode
        ref_tau = self.group_mean_decrements[0]
        self.attenuation_curves = {}

        for g_idx, averager in self.averagers.items():
            if g_idx == 0: continue

            target_flux = averager.average_flux_masked
            # 获取 Flux Error (SEM)
            target_flux_err = averager.average_flux_err
            if target_flux_err is None:
                # 如果没有误差，给一个极小值避免除以0 (虽不应该发生)
                target_flux_err = np.zeros_like(target_flux) + 1e-18

            target_tau = self.group_mean_decrements[g_idx]
            delta_tau = target_tau - ref_tau

            # --- Q 计算 ---
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = target_flux / ref_flux_used
                ratio[ratio <= 0] = np.nan
                q_curve = -np.log(ratio) / delta_tau

                # --- Error Propagation ---
                # sigma_Q = (1/delta_tau) * (sigma_F / F)
                # 注意：ref_flux_used 视为无误差常量
                q_err = (1.0 / np.abs(delta_tau)) * (target_flux_err / np.abs(target_flux))

            self.attenuation_curves[g_idx] = {
                'wavelength': averager.common_wavelength,
                'Q_lambda': q_curve,
                'Q_err': q_err,      # <--- 新增
                'delta_tau': delta_tau
            }

        # --- 计算 Effective Curve 及其误差 ---
        if self.attenuation_curves:
            q_list = [d['Q_lambda'] for d in self.attenuation_curves.values()]
            err_list = [d['Q_err'] for d in self.attenuation_curves.values()]
            w_list = [d['delta_tau'] for d in self.attenuation_curves.values()]

            # 转换为 Masked Array
            stack_q = np.ma.masked_invalid(np.array(q_list))
            stack_err = np.ma.masked_invalid(np.array(err_list))
            weights = np.array(w_list)

            # 1. 加权平均值
            self.effective_curve = np.ma.average(stack_q, axis=0, weights=weights).filled(np.nan)

            # 2. 误差合成
            # 公式: sigma_eff = sqrt( sum( (w_i * sigma_i)^2 ) ) / sum(w_i)
            # 注意：这是假设各组误差相互独立
            w_sum = np.sum(weights)
            weighted_var_sum = np.sum((stack_err * weights[:, np.newaxis])**2, axis=0)
            self.effective_curve_err = (np.sqrt(weighted_var_sum) / w_sum).filled(np.nan) # <--- 新增结果

            self.effective_curve_wave = list(self.attenuation_curves.values())[0]['wavelength']
            print(f"Computed effective curve and errors using {mode} reference.")
        else:
            print("No groups available.")

    # =========================================================================
    # 3. 拟合与推导 (Analysis)
    # =========================================================================
    def fit_polynomial(self, order=3):
        """
        对 Effective Curve 进行多项式拟合 (vs 1/lambda)。

        改进点：
        如果存在误差数组 (effective_curve_err)，则执行加权最小二乘拟合 (Weighted Least Squares)。
        权重设为 w = 1 / sigma。
        """
        if self.effective_curve is None:
            raise ValueError("Run compute_attenuation_curves first.")

        # 1. 准备数据
        wave_ang = self.effective_curve_wave
        y_data = self.effective_curve
        y_err = self.effective_curve_err

        # 转换为 1/μm
        wave_microns = wave_ang / 10000.0
        x_inverse = 1.0 / wave_microns

        # 2. 构建 Mask (剔除 NaN, Inf)
        # 基础 Mask：x 和 y 必须有效
        valid_mask = np.isfinite(y_data) & np.isfinite(x_inverse)

        # 3. 拟合逻辑
        if y_err is not None:
            # 如果有误差，还需要剔除误差无效或为0的点
            valid_mask = valid_mask & np.isfinite(y_err) & (y_err > 0)

            x_fit = x_inverse[valid_mask]
            y_fit = y_data[valid_mask]
            err_fit = y_err[valid_mask]

            # 定义权重: w = 1/sigma
            weights = 1.0 / err_fit

            print(f"Fitting polynomial (Weighted Least Squares) with {len(x_fit)} points...")

            # cov=True 会返回协方差矩阵，可以用来估算系数的误差
            coeffs, cov_matrix = np.polyfit(x_fit, y_fit, order, w=weights, cov=True)

            # 计算系数的标准误差 (diag(cov) 的平方根)
            perr = np.sqrt(np.diag(cov_matrix))
            print("Polynomial Coefficients errors:", perr)

        else:
            # 如果没有误差，执行普通最小二乘法
            print("Fitting polynomial (Ordinary Least Squares, no errors)...")
            x_fit = x_inverse[valid_mask]
            y_fit = y_data[valid_mask]
            coeffs = np.polyfit(x_fit, y_fit, order)

        self.poly_coeffs = coeffs
        print("Polynomial Fit Coefficients (highest power first):", self.poly_coeffs)

        return self.poly_coeffs

    def _poly_shape(self, x):
        """根据拟合系数计算多项式值"""
        return np.polyval(self.poly_coeffs, x)

    def _poly_shape_cubic_only(self, x):
        """仅包含高阶项 (去掉常数项)，用于推导形状"""
        # coeffs 顺序: [c3, c2, c1, c0]
        c3, c2, c1 = self.poly_coeffs[0], self.poly_coeffs[1], self.poly_coeffs[2]
        return c3 * x**3 + c2 * x**2 + c1 * x

    def derive_k_curve_parameters(self, intercept_val=None):
        """
        推导归一化因子 S 和 Rv。

        Parameters
        ----------
        intercept_val : float, optional
            设定无穷远处的截距值 (例如 Calzetti 曲线在 1/lambda=0 处的推断值)。
            如果为 None，则假设 k(0) = 0 (即只使用多项式的高阶项进行归一化)。
        """
        if self.poly_coeffs is None:
            self.fit_polynomial()

        x_B = 1.0 / 0.44
        x_V = 1.0 / 0.55

        # 如果指定了截距，我们需要调整公式 k(x) = S * P_cubic(x) + intercept
        # 归一化条件 k(B) - k(V) = 1
        # => S * (P(B) - P(V)) = 1  (截距相减抵消)

        val_B = self._poly_shape_cubic_only(x_B)
        val_V = self._poly_shape_cubic_only(x_V)

        self.scaling_factor = 1.0 / (val_B - val_V)

        if intercept_val is not None:
            self.intercept_val = intercept_val
        else:
            self.intercept_val = 0.0 # 或者基于拟合的 c0，取决于物理假设

        # 计算 Rv = k(V)
        k_V = self.scaling_factor * val_V + self.intercept_val
        self.derived_Rv = k_V

        print(f"Derived Parameters: S={self.scaling_factor:.4f}, Intercept={self.intercept_val}, Rv={self.derived_Rv:.3f}")

    # =========================================================================
    # 4. 绘图方法 (Visualization)
    # =========================================================================

    def plot_fit_vs_calzetti(self):
        """绘制图表 1：Effective Q_lambda 数据 vs 多项式拟合 vs Calzetti"""
        if self.poly_coeffs is None: self.fit_polynomial()

        wave = self.effective_curve_wave
        x = 10000.0 / wave
        y_fit = self._poly_shape(x)

        fig, ax = plt.subplots()

        # 原始数据
        ax.plot(wave, self.effective_curve, color='blue', lw=3, alpha=0.4,
                label=f'Calculated Curve ({self.curve_mode} ref)')
        ax.errorbar(wave, self.effective_curve, yerr=self.effective_curve_err,
                    fmt='o', color='blue', alpha=0.2, markersize=4, capsize=2)

        # 拟合曲线
        ax.plot(wave, y_fit, color='black', lw=3, ls='--',
                label=r'Polynomial Fit ($x=1/\lambda$)')

        # Calzetti 参考
        calz = self.get_calzetti_law(wave)
        ax.plot(wave, calz, color='red', lw=3, label='Calzetti (2000)')

        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'$Q_\lambda$ (Attenuation Curve)')
        ax.set_title(r'Effective Attenuation Curve: Fit vs Calzetti')
        ax.set_xlim(1500, 8000)
        ax.set_ylim(-2, 7)
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.show()

    def plot_derived_k_curve(self):
        """绘制图表 2：推导出的归一化 Extinction Curve k(lambda)"""
        if self.scaling_factor is None: self.derive_k_curve_parameters()

        wave = np.linspace(1500, 8000, 1000)
        x = 10000.0 / wave

        # 计算 k_curve
        shape_val = self._poly_shape_cubic_only(x)
        k_curve = self.scaling_factor * shape_val + self.intercept_val

        # Calzetti 参考 (Rv=4.05)
        k_calz = 2.659 * self.get_calzetti_law(wave, Rv=4.05) + 4.05

        fig, ax = plt.subplots()
        ax.plot(wave, k_curve, color='blue', lw=4,
                label=f'Derived Curve ($R_V={self.derived_Rv:.2f}$)')
        ax.plot(wave, k_calz, color='red', lw=3, ls='--',
                label='Calzetti (2000) ($R_V=4.05$)')

        # 标注 Rv 点
        ax.scatter([5500], [self.derived_Rv], color='blue', s=200, zorder=10)
        ax.annotate(f'{self.derived_Rv:.2f}', xy=(5500, self.derived_Rv),
                    xytext=(5600, self.derived_Rv-0.5), fontsize=24, color='blue')

        # 辅助线
        ax.axvline(5500, color='gray', ls=':', alpha=0.5)
        ax.axvline(4400, color='gray', ls=':', alpha=0.5)
        ax.text(4400, max(k_curve)*0.9, 'B', ha='center', color='gray')
        ax.text(5500, max(k_curve)*0.9, 'V', ha='center', color='gray')

        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'$k(\lambda)$ (Normalized Extinction)')
        ax.set_title('Derived Attenuation Curve vs. Calzetti')
        ax.set_xlim(1500, 8000)
        ax.set_ylim(0, max(k_curve.max(), k_calz.max())*1.1)
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.show()


    # def plot_reddening_curve_comparison(self,
    #                                     if_show=True,
    #                                     if_input=False,
    #                                     fig=None,
    #                                     ax=None,
    #                                     curve_label='Derived Curve (Data)',  # <--- 在这里输入你的 Label
    #                                     curve_color='blue',                  # <--- 在这里输入你的颜色
    #                                     draw_background=True):               # <--- 控制是否画背景参考线
    #     """
    #     绘制图表 3：去 Rv 化的红化对比 E(l-V)/E(B-V)

    #     Parameters
    #     ----------
    #     curve_label : str
    #         你的数据曲线在图例中的名称。
    #     curve_color : str
    #         你的数据曲线的颜色。
    #     draw_background : bool
    #         是否绘制 MW/SMC/Calzetti 参考线及坐标轴装饰。
    #         (叠加绘图时，建议第一次设为 True，之后设为 False)
    #     """
    #     if self.scaling_factor is None: self.derive_k_curve_parameters()

    #     if if_input:
    #         if fig is None or ax is None:
    #             raise ValueError("If if_input is True, fig and ax must be provided.")
    #     else:
    #         fig, ax = plt.subplots(figsize=(12, 10))
    #         # 如果是新图，强制画背景
    #         draw_background = True

    #     x_grid = np.linspace(0.5, 8.2, 1000) # 1/um

    #     # 1. 计算 Derived Curve 的 E(l-V)/E(B-V)
    #     shape_grid = self._poly_shape_cubic_only(x_grid)
    #     shape_V = self._poly_shape_cubic_only(1.0/0.55)
    #     shape_B = self._poly_shape_cubic_only(1.0/0.44)
    #     y_derived = (shape_grid - shape_V) / (shape_B - shape_V)

    #     # 2. 绘制背景参考线 (仅当 draw_background=True 时)
    #     if draw_background:
    #         y_calz = self._get_normalized_reddening(x_grid, 'calzetti')
    #         y_mw = self._get_normalized_reddening(x_grid, 'mw')
    #         y_smc = self._get_normalized_reddening(x_grid, 'smc')

    #         ax.plot(x_grid, y_mw, color='green', ls=':', lw=2, label='Milky Way (CCM89)')
    #         ax.plot(x_grid, y_smc, color='purple', ls='-.', lw=2, label='SMC (Pei92)')
    #         ax.plot(x_grid, y_calz, color='black', ls='--', lw=3, label='Calzetti (2000)')

    #         # 绘制辅助线和文本
    #         x_V, x_B = 1.0/0.55, 1.0/0.44
    #         ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    #         ax.axvline(x_V, color='gray', linestyle='-', alpha=0.3)
    #         ax.scatter([x_V], [0], color='black', s=80, zorder=10)
    #         ax.text(x_V, 0.5, 'V (5500$\AA$)\nNormalized to 0', rotation=90, color='gray', fontsize=14, ha='right')

    #         ax.axhline(1, color='gray', linestyle=':', alpha=0.3)
    #         ax.axvline(x_B, color='gray', linestyle=':', alpha=0.3)
    #         ax.scatter([x_B], [1], color='black', s=80, zorder=10)
    #         ax.text(x_B, 1.5, 'B (4400$\AA$)\nNormalized to 1', rotation=90, color='gray', fontsize=14, ha='right')

    #         # 设置轴标签和标题
    #         ax.set_xlabel(r'Inverse Wavelength $\lambda^{-1}$ ($\mu m^{-1}$)', fontsize=20)
    #         ax.set_ylabel(r'$E(\lambda - V) / E(B - V)$', fontsize=20)
    #         ax.set_title(r'Reddening Curves Comparison (Independent of $R_V$)', fontsize=24)
    #         ax.set_xlim(0.5, 8.2)
    #         ax.set_ylim(-2, 12)

    #         # 顶部波长轴
    #         ax2 = ax.twiny()
    #         ticks = [10000, 5500, 4400, 3000, 2000, 1500]
    #         ax2.set_xlim(ax.get_xlim())
    #         ax2.set_xticks([10000.0/t for t in ticks])
    #         ax2.set_xticklabels([str(t) for t in ticks], fontsize=14)
    #         ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)

    #         ax.grid(True, alpha=0.2)

    #     # 3. 绘制你的数据 (使用自定义的 label 和 color)
    #     ax.plot(x_grid, y_derived, color=curve_color, lw=4, label=curve_label)

    #     # 4. 图例去重处理 (防止多次调用导致图例重复)
    #     handles, labels = ax.get_legend_handles_labels()
    #     # 使用字典保持插入顺序并去重
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=16)

    #     plt.tight_layout()
    #     if if_show:
    #         plt.show()
    #     return fig, ax

    def plot_reddening_curve_comparison(self,
                                        if_show=True,
                                        if_input=False,
                                        fig=None,
                                        ax=None,
                                        curve_label='Derived Curve (Data)',
                                        curve_color='blue',
                                        draw_background=True,
                                        plot_mcmc_samples=False,  # <--- 新接口
                                        n_samples=100):           # <--- 新接口
        """
        绘制去 Rv 化的红化对比图，支持叠加 MCMC 样本线。
        """
        if self.scaling_factor is None: self.derive_k_curve_parameters()

        if if_input:
            if fig is None or ax is None:
                raise ValueError("If if_input is True, fig and ax must be provided.")
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            draw_background = True

        x_grid = np.linspace(0.5, 8.2, 1000) # 1/um

        # ==========================================
        # 1. 准备 MCMC 样本线 (如果在 MCMC 之后)
        # ==========================================
        sample_lines_y = []
        if plot_mcmc_samples and hasattr(self, 'mcmc_samples'):
            # 定义 B 和 V 的频率
            x_B, x_V = 1.0/0.44, 1.0/0.55

            # 随机抽取样本
            n_total = len(self.mcmc_samples)
            inds = np.random.randint(n_total, size=n_samples)

            for ind in inds:
                coeffs = self.mcmc_samples[ind]
                # 提取形状部分 (假设只用高阶项决定形状，如果拟合包含c0但假设c0不影响形状)
                # 注意：这里的逻辑必须与 _poly_shape_cubic_only 保持一致
                # theta = [c3, c2, c1, c0]
                c3, c2, c1 = coeffs[0], coeffs[1], coeffs[2]

                # 局部函数计算该样本的形状值
                def shape_func(x):
                    return c3 * x**3 + c2 * x**2 + c1 * x

                val_grid = shape_func(x_grid)
                val_B = shape_func(x_B)
                val_V = shape_func(x_V)

                # 计算归一化曲线
                y_sample = (val_grid - val_V) / (val_B - val_V)
                sample_lines_y.append(y_sample)

        # ==========================================
        # 2. 计算最佳拟合曲线 (Best Fit)
        # ==========================================
        shape_grid = self._poly_shape_cubic_only(x_grid)
        shape_V = self._poly_shape_cubic_only(1.0/0.55)
        shape_B = self._poly_shape_cubic_only(1.0/0.44)
        y_derived = (shape_grid - shape_V) / (shape_B - shape_V)

        # ==========================================
        # 3. 绘制背景参考线
        # ==========================================
        if draw_background:
            y_calz = self._get_normalized_reddening(x_grid, 'calzetti')
            y_mw = self._get_normalized_reddening(x_grid, 'mw')
            y_smc = self._get_normalized_reddening(x_grid, 'smc')

            ax.plot(x_grid, y_mw, color='green', ls=':', lw=2, label='Milky Way (CCM89)')
            ax.plot(x_grid, y_smc, color='purple', ls='-.', lw=2, label='SMC (Pei92)')
            ax.plot(x_grid, y_calz, color='black', ls='--', lw=3, label='Calzetti (2000)')

            # 辅助线
            x_V, x_B = 1.0/0.55, 1.0/0.44
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
            ax.axvline(x_V, color='gray', linestyle='-', alpha=0.3)
            ax.scatter([x_V], [0], color='black', s=80, zorder=10)
            ax.text(x_V, 0.5, 'V (5500$\AA$)\nNormalized to 0', rotation=90, color='gray', fontsize=14, ha='right')

            ax.axhline(1, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x_B, color='gray', linestyle=':', alpha=0.3)
            ax.scatter([x_B], [1], color='black', s=80, zorder=10)
            ax.text(x_B, 1.5, 'B (4400$\AA$)\nNormalized to 1', rotation=90, color='gray', fontsize=14, ha='right')

            ax.set_xlabel(r'Inverse Wavelength $\lambda^{-1}$ ($\mu m^{-1}$)', fontsize=20)
            ax.set_ylabel(r'$E(\lambda - V) / E(B - V)$', fontsize=20)
            ax.set_title(r'Reddening Curves Comparison (Independent of $R_V$)', fontsize=24)
            ax.set_xlim(0.5, 8.2)
            ax.set_ylim(-2, 12)

            # 顶部波长轴
            ax2 = ax.twiny()
            ticks = [10000, 5500, 4400, 3000, 2000, 1500]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks([10000.0/t for t in ticks])
            ax2.set_xticklabels([str(t) for t in ticks], fontsize=14)
            ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)
            ax.grid(True, alpha=0.2)

        # ==========================================
        # 4. 绘制 MCMC 样本线 (在最佳拟合线下面绘制)
        # ==========================================
        if sample_lines_y:
            for y_samp in sample_lines_y:
                # 使用与主曲线相同的颜色，但透明度极低
                ax.plot(x_grid, y_samp, color=curve_color, alpha=0.05, lw=1)

        # ==========================================
        # 5. 绘制最佳拟合线
        # ==========================================
        ax.plot(x_grid, y_derived, color=curve_color, lw=4, label=curve_label)

        # 图例去重
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=16)

        plt.tight_layout()
        if if_show:
            plt.show()
        return fig, ax

    # =========================================================================
    # 5. 静态辅助方法 (物理定律)
    # =========================================================================

    @staticmethod
    def get_calzetti_law(wavelength_angstrom, Rv=4.05):
        """返回 Calzetti k(lambda) 其中的形状部分 (normalized so roughly Q) 或完整公式"""
        # 注意：这里返回的是用于 Q_lambda 对比的形状 (-2.156... 那部分)
        # 如果需要完整 k(lambda)，需要乘以 2.659 并加 Rv
        microns = wavelength_angstrom / 10000.0
        k_lambda = np.zeros_like(microns)

        mask1 = (microns >= 0.12) & (microns < 0.63)
        w1 = microns[mask1]
        k_lambda[mask1] = (-2.156 + 1.509/w1 - 0.198/(w1**2) + 0.011/(w1**3))

        mask2 = (microns >= 0.63) & (microns <= 2.20)
        w2 = microns[mask2]
        k_lambda[mask2] = (-1.857 + 1.040/w2)

        return k_lambda
    @staticmethod
    def _get_normalized_reddening(x_grid, type_name):
        """
        Calculate the reddening curve normalized as E(lambda-V) / E(B-V).
        Formula: (A_lambda - A_V) / (A_B - A_V)

        This shape is independent of Rv for the target (though internal models use standard Rvs).
        """
        x_grid = np.array(x_grid)

        # --- Internal Helper: MW (Cardelli, Clayton, & Mathis 1989) ---
        def get_mw_ccm89_Av(x, Rv=3.1):
            x = np.array(x)
            a = np.zeros_like(x)
            b = np.zeros_like(x)

            # 1. Infrared (0.3 <= x < 1.1)
            mask_ir = (x >= 0.3) & (x < 1.1)
            if np.any(mask_ir):
                a[mask_ir] = 0.574 * x[mask_ir]**1.61
                b[mask_ir] = -0.527 * x[mask_ir]**1.61

            # 2. Optical/NIR (1.1 <= x < 3.3)
            mask_opt = (x >= 1.1) & (x < 3.3)
            if np.any(mask_opt):
                y = x[mask_opt] - 1.82
                a[mask_opt] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
                b[mask_opt] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

            # 3. UV (3.3 <= x <= 8)
            mask_uv = (x >= 3.3)
            if np.any(mask_uv):
                X = x[mask_uv]
                F_a = -0.04473 * (X - 5.9)**2 - 0.009779 * (X - 5.9)**3
                F_b = 0.2130 * (X - 5.9)**2 + 0.1207 * (X - 5.9)**3
                # Only apply F_a, F_b for x >= 5.9
                low_uv_mask = X < 5.9
                F_a[low_uv_mask] = 0
                F_b[low_uv_mask] = 0

                a[mask_uv] = 1.752 - 0.316*X - 0.104/((X - 4.67)**2 + 0.341) + F_a
                b[mask_uv] = -3.090 + 1.825*X + 1.206/((X - 4.67)**2 + 0.263) + F_b

            return a + b / Rv

        # --- Internal Helper: SMC (Pei 1992) ---
        def get_smc_pei92_Av(x):
            a_params = [185, 27, 0.005, 0.010, 0.012, 0.030]
            w_params = [0.042, 0.08, 0.22, 9.7, 18, 25] # lambda_i in microns
            b_params = [90, 5.5, -1.95, -1.95, -1.80, 0.0]
            n_params = [2.0, 4.0, 2.0, 2.0, 2.0, 2.0]

            lam = 1.0 / x # microns
            ext_sum = np.zeros_like(x)

            for i in range(6):
                term = a_params[i] / ((lam/w_params[i])**n_params[i] + (w_params[i]/lam)**n_params[i] + b_params[i])
                ext_sum += term

            # Calculate normalization factor at V-band (0.55 um => x=1.818)
            lam_V = 0.55
            val_V = 0
            for i in range(6):
                val_V += a_params[i] / ((lam_V/w_params[i])**n_params[i] + (w_params[i]/lam_V)**n_params[i] + b_params[i])

            return ext_sum / val_V

        # --- Internal Helper: Calzetti 2000 ---
        def get_calzetti_curve_Av(x, Rv=4.05):
            lam_um = 1.0 / x
            k = np.zeros_like(x)

            # Range 1: 0.12um to 0.63um
            mask1 = (lam_um >= 0.12) & (lam_um < 0.63)
            if np.any(mask1):
                l = lam_um[mask1]
                k[mask1] = 2.659 * (-2.156 + 1.509/l - 0.198/l**2 + 0.011/l**3) + Rv

            # Range 2: 0.63um to 2.2um
            mask2 = (lam_um >= 0.63) & (lam_um <= 2.2)
            if np.any(mask2):
                l = lam_um[mask2]
                k[mask2] = 2.659 * (-1.857 + 1.040/l) + Rv

            # Returns A_lambda / A_V
            return k / Rv

        # --- Selector Logic ---
        if type_name == 'mw':
            y_vals = get_mw_ccm89_Av(x_grid, Rv=3.1)
        elif type_name == 'smc':
            y_vals = get_smc_pei92_Av(x_grid)
        elif type_name == 'calzetti':
            y_vals = get_calzetti_curve_Av(x_grid, Rv=4.05)
        else:
            return np.zeros_like(x_grid)

        # --- Normalization ---
        # Calculate B and V band values for the chosen model
        x_B = 1.0 / 0.44  # B-band inverse micron
        x_V = 1.0 / 0.55  # V-band inverse micron

        # Helper to compute single points without array shape issues
        def get_pt(x_pt):
            if type_name == 'mw': return get_mw_ccm89_Av(np.array([x_pt]), Rv=3.1)[0]
            if type_name == 'smc': return get_smc_pei92_Av(np.array([x_pt]))[0]
            if type_name == 'calzetti': return get_calzetti_curve_Av(np.array([x_pt]), Rv=4.05)[0]
            return 0.0

        y_B = get_pt(x_B)
        y_V = get_pt(x_V)

        # Formula: (A_lambda - A_V) / (A_B - A_V)
        # y_vals corresponds to A_lambda / A_V
        # Therefore: (y_vals * Av - Av) / (y_B * Av - Av) = (y_vals - 1) / (y_B - 1)
        # Note: If y_V is exactly 1.0, this simplifies to (y_vals - 1)/(y_B - 1).
        # We use y_V variable to be numerically safe.

        return (y_vals - y_V) / (y_B - y_V)

    def plot_extinction_curves_comparison(self):
            """
            绘制图表 3：Extinction Curves Comparison (A_lambda / A_V vs 1/lambda)

            对比以下曲线：
            1. Milky Way (Cardelli et al. 1989), Rv=3.1
            2. SMC (Pei 1992)
            3. Calzetti (2000), Rv=4.05
            4. Derived Curve (假设 k(0)=0)
            5. Derived Curve (假设 k(0)=intercept)
            """

            # 确保前置计算已完成
            if self.poly_coeffs is None:
                self.fit_polynomial()
            if self.scaling_factor is None:
                self.derive_k_curve_parameters()

            # 定义 X 轴网格: 1/lambda [um^-1]
            x_grid = np.linspace(0.5, 8.2, 1000)

            # =========================================================
            # 内部核心函数：计算参考模型的 A_lambda / A_V
            # =========================================================
            def get_ref_Av(x, model):
                x = np.array(x)

                # --- 1. Calzetti (2000) Starburst ---
                if model == 'calzetti':
                    Rv = 4.05
                    lam = 1.0 / x  # microns
                    k = np.zeros_like(x)

                    # Range 1: 0.12 to 0.63 microns
                    mask1 = (lam >= 0.12) & (lam < 0.63)
                    if np.any(mask1):
                        l = lam[mask1]
                        # k(lambda) formula
                        k[mask1] = 2.659 * (-2.156 + 1.509/l - 0.198/l**2 + 0.011/l**3) + Rv

                    # Range 2: 0.63 to 2.20 microns
                    mask2 = (lam >= 0.63) & (lam <= 2.2)
                    if np.any(mask2):
                        l = lam[mask2]
                        k[mask2] = 2.659 * (-1.857 + 1.040/l) + Rv

                    # Return A_lambda / A_V = k(lambda) / Rv
                    return k / Rv

                # --- 2. Milky Way (CCM 1989) ---
                elif model == 'mw':
                    Rv = 3.1
                    a = np.zeros_like(x)
                    b = np.zeros_like(x)

                    # Infrared (0.3 <= x < 1.1)
                    mask_ir = (x >= 0.3) & (x < 1.1)
                    if np.any(mask_ir):
                        xi = x[mask_ir]
                        a[mask_ir] = 0.574 * xi**1.61
                        b[mask_ir] = -0.527 * xi**1.61

                    # Optical/NIR (1.1 <= x < 3.3)
                    mask_opt = (x >= 1.1) & (x < 3.3)
                    if np.any(mask_opt):
                        y = x[mask_opt] - 1.82
                        a[mask_opt] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
                        b[mask_opt] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

                    # UV (3.3 <= x <= 8)
                    mask_uv = (x >= 3.3)
                    if np.any(mask_uv):
                        X = x[mask_uv]
                        Fa = -0.04473 * (X - 5.9)**2 - 0.009779 * (X - 5.9)**3
                        Fb = 0.2130 * (X - 5.9)**2 + 0.1207 * (X - 5.9)**3

                        # Fa, Fb only active above 5.9 um^-1
                        low_uv = X < 5.9
                        Fa[low_uv] = 0
                        Fb[low_uv] = 0

                        a[mask_uv] = 1.752 - 0.316*X - 0.104/((X - 4.67)**2 + 0.341) + Fa
                        b[mask_uv] = -3.090 + 1.825*X + 1.206/((X - 4.67)**2 + 0.263) + Fb

                    # A_lambda / A_V = a(x) + b(x) / Rv
                    return a + b / Rv

                # --- 3. SMC (Pei 1992) ---
                elif model == 'smc':
                    # Coefficients from Pei (1992) Table 2 (SMC column)
                    a_params = [185, 27, 0.005, 0.010, 0.012, 0.030]
                    w_params = [0.042, 0.08, 0.22, 9.7, 18, 25] # microns
                    b_params = [90, 5.5, -1.95, -1.95, -1.80, 0.0]
                    n_params = [2.0, 4.0, 2.0, 2.0, 2.0, 2.0]

                    lam = 1.0 / x # microns
                    ext_sum = np.zeros_like(x)

                    # Sum the 6 terms
                    for i in range(6):
                        term = a_params[i] / ((lam/w_params[i])**n_params[i] + (w_params[i]/lam)**n_params[i] + b_params[i])
                        ext_sum += term

                    # Normalize at V-band (0.55 um -> x=1.818)
                    lam_V = 0.55
                    val_V = 0
                    for i in range(6):
                        val_V += a_params[i] / ((lam_V/w_params[i])**n_params[i] + (w_params[i]/lam_V)**n_params[i] + b_params[i])

                    # Return Normalized A_lambda / A_V
                    return ext_sum / val_V

                return np.zeros_like(x)

            # =========================================================
            # 2. 计算数据
            # =========================================================

            # A. 参考曲线
            y_calzetti = get_ref_Av(x_grid, 'calzetti')
            y_mw = get_ref_Av(x_grid, 'mw')
            y_smc = get_ref_Av(x_grid, 'smc')

            # B. 你的 Derived Curves (根据之前计算的多项式和归一化因子)
            # k(x) = S * P_cubic(x) + intercept
            # A/Av = k(x) / Rv

            shape_grid = self._poly_shape_cubic_only(x_grid)
            shape_V = self._poly_shape_cubic_only(1.0/0.55) # V-band value of shape

            # --- Case 1: 假设 k(0) = 0 ---
            # 此时 intercept = 0
            # S_zero 由 k(B)-k(V)=1 决定，这已经是 self.scaling_factor 了
            # Rv_zero = k(V) = S * shape(V)
            k_zero_curve = self.scaling_factor * shape_grid
            Rv_zero = self.scaling_factor * shape_V
            y_derived_zero = k_zero_curve / Rv_zero

            # --- Case 2: 假设 k(0) = intercept (Shifted) ---
            # k(x) = S * shape(x) + intercept
            # Rv_shifted = k(V) = S * shape(V) + intercept
            k_shifted_curve = self.scaling_factor * shape_grid + self.intercept_val
            Rv_shifted = self.scaling_factor * shape_V + self.intercept_val # 应该等于 self.derived_Rv
            y_derived_shifted = k_shifted_curve / Rv_shifted

            # =========================================================
            # 3. 绘图
            # =========================================================
            fig, ax = plt.subplots(figsize=(14, 10))

            # 绘制参考模型
            ax.plot(x_grid, y_mw, color='green', ls=':', lw=2, label='Milky Way (CCM89, $R_V=3.1$)')
            ax.plot(x_grid, y_smc, color='purple', ls='-.', lw=2, label='SMC (Pei92)')
            ax.plot(x_grid, y_calzetti, color='black', ls='--', lw=3, label='Calzetti (2000, $R_V=4.05$)')

            # 绘制推导出的曲线
            ax.plot(x_grid, y_derived_zero, color='dodgerblue', lw=3, alpha=0.8,
                    label=f'Derived (Assume $k(0)=0$, $R_V={Rv_zero:.2f}$)')

            ax.plot(x_grid, y_derived_shifted, color='firebrick', lw=3,
                    label=f'Derived (Assume $k(0)={self.intercept_val:.2f}$, $R_V={Rv_shifted:.2f}$)')

            # 辅助标记 V 和 B 波段
            x_V = 1.0/0.55
            x_B = 1.0/0.44

            ax.axvline(x_V, color='gray', ls='-', alpha=0.3)
            ax.scatter([x_V]*2, [1]*2, color='gray', s=50, zorder=10) # A/Av 在 V 波段恒为 1
            ax.text(x_V, 0.2, 'V (5500$\AA$)', rotation=90, color='gray', fontsize=16, ha='right')

            ax.axvline(x_B, color='gray', ls='-', alpha=0.3)
            ax.text(x_B, 0.2, 'B (4400$\AA$)', rotation=90, color='gray', fontsize=16, ha='right')

            # 坐标轴设置
            ax.set_xlabel(r'Inverse Wavelength $\lambda^{-1}$ ($\mu m^{-1}$)', fontsize=20)
            ax.set_ylabel(r'$A_\lambda / A_V$', fontsize=20)
            ax.set_title('Extinction Curves Comparison', fontsize=24)
            ax.set_xlim(0.5, 8.2)
            ax.set_ylim(0, 9) # 根据 UV 上升幅度调整

            # 顶部波长刻度
            ax2 = ax.twiny()
            ticks = [10000, 5500, 4400, 3000, 2000, 1500, 1250]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks([10000.0/t for t in ticks])
            ax2.set_xticklabels([str(t) for t in ticks], fontsize=14)
            ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)

            ax.legend(loc='upper left', fontsize=16)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.show()

    def run_mcmc_fitting(self, nwalkers=32, nsteps=2000, poly_order=3):
        """
        使用 MCMC (emcee) 拟合，并强制 UV 端单调递增 (Derivative >= 0)。
        """
        import emcee

        if self.effective_curve is None or self.effective_curve_err is None:
            raise ValueError("Data or Error missing. Run compute_attenuation_curves first.")

        # 1. 准备数据 (x = 1/lambda)
        wave_um = self.effective_curve_wave / 10000.0
        x_data = 1.0 / wave_um
        y_data = self.effective_curve
        y_err = self.effective_curve_err

        # 剔除 NaN 和 Inf
        mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(y_err) & (y_err > 0)
        x_fit, y_fit, err_fit = x_data[mask], y_data[mask], y_err[mask]

        # -----------------------------------------------------------------
        # 定义内部函数
        # -----------------------------------------------------------------

        # 模型: c3*x^3 + c2*x^2 + c1*x + c0
        def model(theta, x):
            return np.polyval(theta, x)

        def log_likelihood(theta, x, y, yerr):
            model_y = model(theta, x)
            sigma2 = yerr ** 2
            return -0.5 * np.sum((y - model_y) ** 2 / sigma2 + np.log(sigma2))

        def log_prior(theta):
            # 1. 基础参数范围限制 (防止跑飞)
            # theta 顺序是 [c3, c2, c1, c0] (最高次幂在前)
            c3, c2, c1, c0 = theta
            if not (-10 < c3 < 10 and -20 < c2 < 20 and -20 < c1 < 20 and -10 < c0 < 10):
                return -np.inf

            # 2. 【核心修改】强制 UV 端单调递增 (Derivative >= 0)
            # 我们检查 x 在 [2.5, 8.5] 范围内 (即 UV 到 远UV)
            # 多项式 P(x) = c3*x^3 + c2*x^2 + c1*x + c0
            # 导数 P'(x) = 3*c3*x^2 + 2*c2*x + c1

            x_check = np.linspace(2.5, 8.5, 20) # 检查点的范围
            deriv = 3 * c3 * (x_check**2) + 2 * c2 * x_check + c1

            # 如果在检查范围内任何一点导数小于 0 (曲线下降)，则拒绝该参数
            if np.any(deriv < 0):
                return -np.inf

            return 0.0

        def log_probability(theta, x, y, yerr):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, x, y, yerr)

        # -----------------------------------------------------------------
        # 运行 MCMC
        # -----------------------------------------------------------------

        # 初始猜测
        initial_coeffs = np.polyfit(x_fit, y_fit, poly_order)
        ndim = len(initial_coeffs)
        pos = initial_coeffs + 1e-4 * np.random.randn(nwalkers, ndim)

        print(f"Running MCMC with {nwalkers} walkers for {nsteps} steps (Monotonicity Constrained)...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x_fit, y_fit, err_fit))
        sampler.run_mcmc(pos, nsteps, progress=True)

        discard = int(nsteps * 0.25)
        self.mcmc_samples = sampler.get_chain(discard=discard, thin=15, flat=True)

        # 更新最佳系数
        self.poly_coeffs = np.median(self.mcmc_samples, axis=0)
        print("MCMC Best Fit Coeffs:", self.poly_coeffs)

        return self.mcmc_samples

    def plot_mcmc_results(self, plot_samples=True, n_samples=100):
        """
        绘制 MCMC 拟合结果：包含数据点误差棒、最佳拟合线、误差带以及(可选)随机样本猜测线。

        Parameters
        ----------
        plot_samples : bool
            是否画出从 MCMC 链中随机抽取的样本线 (Spaghetti plot)。
        n_samples : int
            画多少条样本线。
        """
        if not hasattr(self, 'mcmc_samples'):
            print("Please run run_mcmc_fitting first.")
            return

        # 定义绘图网格
        x_grid = 1.0 / (np.linspace(1500, 8000, 500) / 10000.0) # 1/um
        wave_grid = 10000.0 / x_grid

        # 1. 抽取随机样本
        # 如果样本数足够，随机抽取；否则全部使用
        n_total = len(self.mcmc_samples)
        if n_samples > n_total: n_samples = n_total
        inds = np.random.randint(n_total, size=n_samples)

        sample_curves = []
        for ind in inds:
            sample = self.mcmc_samples[ind]
            # 计算该样本对应的曲线值
            sample_curves.append(np.polyval(sample, x_grid))

        sample_curves = np.array(sample_curves)

        # 计算统计量用于误差带
        lower_bound = np.percentile(sample_curves, 2.5, axis=0)
        upper_bound = np.percentile(sample_curves, 97.5, axis=0)

        # 最佳拟合 (使用所有样本的中位数系数，或者当前存储的 poly_coeffs)
        best_fit_curve = self._poly_shape(x_grid)

        # 2. 绘图
        fig, ax = plt.subplots(figsize=(12, 8))

        # A. 绘制随机样本猜测线 (The "Guesses")
        if plot_samples:
            # 同样是画 x_grid 对应的曲线
            # 颜色使用 alpha=0.1 变淡
            for i in range(len(sample_curves)):
                ax.plot(wave_grid, sample_curves[i], color='dodgerblue', alpha=0.05, lw=1)

        # B. 原始数据带误差棒
        wave_data = self.effective_curve_wave
        y_data = self.effective_curve
        y_err = self.effective_curve_err
        step = 10
        ax.errorbar(wave_data[::step], y_data[::step], yerr=y_err[::step],
                    fmt='o', color='black', alpha=0.6, label='Data (with SEM)', markersize=4, capsize=0)

        # C. 拟合带 (95% CI)
        # 如果画了 sample lines，fill_between 可以选择不画或者颜色淡一点
        ax.fill_between(wave_grid, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Credible Interval')

        # D. 最佳拟合线
        ax.plot(wave_grid, best_fit_curve, color='blue', lw=2, label='MCMC Best Fit')

        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'$Q_\lambda$')
        ax.set_title('MCMC Polynomial Fit with Posterior Samples')
        ax.set_xlim(1500, 8000)
        ax.set_ylim(-2, 8)

        # 处理 Legend，避免重复
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        plt.show()
