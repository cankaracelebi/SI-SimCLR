import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


    
features_dict = {    
    "Chebyshev Fourier Statistics": [
        f"chebyshev_fourier_stats-num_coeffs_400-num_bins_32-hist_bin{i}" for i in range(32)
    ],
    "Chebyshev Statistics": [
        f"chebyshev_stats-num_coeffs_400-num_bins_32-hist_bin{i}" for i in range(32)
    ],
    "Edge Statistics": [
        "edge_stats-direction_hist_bin0", "edge_stats-direction_hist_bin1",
        "edge_stats-direction_hist_bin2", "edge_stats-direction_hist_bin3",
        "edge_stats-direction_hist_bin4", "edge_stats-direction_hist_bin5",
        "edge_stats-direction_hist_bin6", "edge_stats-direction_hist_bin7",
        "edge_stats-direction_hist_diff_bin0", "edge_stats-direction_hist_diff_bin1",
        "edge_stats-direction_hist_diff_bin2", "edge_stats-direction_hist_diff_bin3",
        "edge_stats-direction_homogeneity", "edge_stats-direction_mean",
        "edge_stats-direction_median", "edge_stats-direction_variance",
        "edge_stats-edge_density", "edge_stats-magnitude_hist_bin0",
        "edge_stats-magnitude_hist_bin1", "edge_stats-magnitude_hist_bin2",
        "edge_stats-magnitude_hist_bin3", "edge_stats-magnitude_hist_bin4",
        "edge_stats-magnitude_hist_bin5", "edge_stats-magnitude_hist_bin6",
        "edge_stats-magnitude_hist_bin7", "edge_stats-magnitude_mean",
        "edge_stats-magnitude_median", "edge_stats-magnitude_variance"
    ],
    "Gabor Filter Features": [
        f"gabor-frequency_{freq}-theta_{theta}-sigma_x_{sigma_x}-sigma_y_{sigma_y}-{stat}"
        for freq in [0.05, 0.25]
        for theta in [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345]
        for sigma_x, sigma_y in [(1.0, 1.0), (3.0, 3.0)]
        for stat in ["mean_imag", "mean_real", "var"]
    ],
    "Haralick Texture Features": [
        f"haralick-distances_{dist}-angles_{angles}-{stat}"
        for dist in [1, 2]
        for angles in [
            "0.0_0.39269908169872414_0.7853981633974483_1.1780972450961724_1.5707963267948966_1.9634954084936207_2.356194490192345_2.748893571891069",
            "0.0_0.7853981633974483_1.5707963267948966_2.356194490192345"
        ]
        for stat in [
            "contrast", "correlation", "dissimilarity", "energy", "entropy", "homogeneity"
        ]
    ],
    "Image Statistics": [
        'image_stats-max', 'image_stats-mean',
        'image_stats-mean_intensity_above_threshold',
        'image_stats-num_pixels_above_threshold', 'image_stats-percent_maximal',
        'image_stats-percent_minimal', 'image_stats-percent_pixels_above_threshold',
        'image_stats-total_intensity', 'image_stats-variance_above_threshold',
        'image_stats-variance',
    ],
    "Intensity Statistics": [
        f"intensity_stats-hist_bin{i}" for i in range(32)
    ] + [
        "intensity_stats-hist_diff_bin0", "intensity_stats-hist_diff_bin1",
        "intensity_stats-hist_diff_bin2", "intensity_stats-hist_diff_bin3",
        "intensity_stats-hist_diff_bin4", "intensity_stats-hist_diff_bin5",
        "intensity_stats-hist_diff_bin6", "intensity_stats-hist_diff_bin7",
        "intensity_stats-hist_homogeneity", "intensity_stats-mean",
        "intensity_stats-median", "intensity_stats-variance"
    ],
    "Statistical Moments": [
        "moments-kurtosis", "moments-mean", "moments-skewness", "moments-variance"
    ],
    "Shannon Entropy": [
        "shannon_entropy-blue", "shannon_entropy-green", "shannon_entropy-red"
    ],
    "Tamura Features": [
        f"tamura-max_k_{k}-num_bins_16-{stat}"
        for k in [2, 3, 4]
        for stat in [
            "coarseness", "contrast", "directionality", 
            "hist_bin0", "hist_bin1", "hist_bin2"
        ]
    ]
}



def create_folds(raw_emb, features, n_s=5, n_grp=None):
    folds = np.full(raw_emb.shape[0], -1)
    
    if n_grp is None:
        skf = StratifiedKFold(n_splits=n_s)
        target = features
    else:
        skf = StratifiedKFold(n_splits=n_s)
        grp = pd.cut(features, n_grp, labels=False)
        target = grp
    
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        folds[v] = fold_no
    
    return folds


