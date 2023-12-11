def limit(lowest: int, current_value: int, maximum_value: int) -> int:
    return min(max(lowest, current_value), maximum_value)
