def limit(lowest: int, current_value: int, maximum_value: int) -> int:
    return min(max(lowest, current_value), maximum_value)

def scale_value(input_value, input_lower, input_upper, output_lower, output_upper):
    # Scale the input value
    scaled_value = ((input_value - input_lower) / (input_upper - input_lower)) * (output_upper - output_lower) + output_lower
    scaled_value = limit(output_lower,scaled_value,output_upper)
    return scaled_value