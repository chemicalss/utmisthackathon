# persistence.py
def check_persistence(region, min_persisted_mm: float = 5.0) -> bool:
    return region.persisted_mm >= min_persisted_mm
