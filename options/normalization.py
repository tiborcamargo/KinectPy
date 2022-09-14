from utils.normalization import normalize_obb, normalize_obb_rotation_translation, translate

normalization_options = {
    'obb_normalization': normalize_obb,
    'obb_rotation_translation': normalize_obb_rotation_translation,
    'translation': translate,
    }
