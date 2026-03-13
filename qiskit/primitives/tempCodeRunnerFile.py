def _memory_array(results: list[list[str]], num_bytes: int) -> NDArray[np.uint8]:
    """Converts the memory data into an array in an unpacked way."""
    lst = []
    for memory in results:
        for i in memory:
            value = int(i, 16)
            required_bytes = (value.bit_length() + 7) // 8
            if required_bytes > num_bytes:
                print("Mismatch:", i, required_bytes, num_bytes)
        