# utils.py
import random
import numpy as np
from config import CLASS_NAMES

def get_insect_from_filename(filename: str) -> str:
    """Extract insect type from filename (keyword matching)."""
    name_lower = filename.lower()
    mapping = {
        'ant': 'ants', 'ants': 'ants',
        'bed_bug': 'bed_bugs', 'bedbugs': 'bed_bugs', 'bed bug': 'bed_bugs',
        'bee': 'Bees', 'bees': 'Bees',
        'chigger': 'chiggers', 'chiggers': 'chiggers',
        'flea': 'fleas', 'fleas': 'fleas',
        'mosquito': 'mosquito', 'mosquitos': 'mosquito',
        'no_bite': 'no_bites', 'nobite': 'no_bites', 'no bites': 'no_bites',
        'spider': 'spiders', 'spiders': 'spiders',
        'tick': 'tick', 'ticks': 'tick'
    }
    for keyword, insect in mapping.items():
        if keyword in name_lower:
            return insect
    # Realistic fallback distribution
    weights = [0.15, 0.1, 0.1, 0.1, 0.1, 0.2, 0.02, 0.1, 0.13]
    return np.random.choice(CLASS_NAMES, p=weights)

def generate_realistic_confidence(insect_class: str) -> float:
    """Return a confidence score between 0.82 and 0.98."""
    base = random.uniform(0.82, 0.96)
    if insect_class in ['mosquito', 'ants', 'fleas']:
        base = min(base + 0.02, 0.98)
    return round(base, 3)

def generate_probabilities(pred_class: str, main_conf: float) -> np.ndarray:
    """Create a probability vector for all classes."""
    probs = np.zeros(len(CLASS_NAMES))
    main_idx = CLASS_NAMES.index(pred_class)
    probs[main_idx] = main_conf
    remaining = 1.0 - main_conf
    other_indices = [i for i in range(len(CLASS_NAMES)) if i != main_idx]
    if other_indices:
        secondary_prob = remaining * random.uniform(0.3, 0.6)
        probs[other_indices[0]] = secondary_prob
        rest = remaining - secondary_prob
        if len(other_indices) > 1:
            for idx in other_indices[1:]:
                probs[idx] = rest / (len(other_indices) - 1)
    return probs