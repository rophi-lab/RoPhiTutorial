import numpy as np

# from lcm_type.vision.rgbd_t import rgbd_t

# Enum mapping from LCM to NumPy dtype
CHANNEL_TYPE_TO_DTYPE = {
    0: np.int8,  # CHANNEL_TYPE_INT8
    1: np.uint8,  # CHANNEL_TYPE_UINT8
    2: np.int16,  # CHANNEL_TYPE_INT16
    3: np.uint16,  # CHANNEL_TYPE_UINT16
    4: np.int32,  # CHANNEL_TYPE_INT32
    5: np.uint32,  # CHANNEL_TYPE_UINT32
    6: np.float32,  # CHANNEL_TYPE_FLOAT32
    7: np.float64,  # CHANNEL_TYPE_FLOAT64
}

DTYPE_TO_CHANNEL_TYPE = {v: k for k, v in CHANNEL_TYPE_TO_DTYPE.items()}


def pack_image_to_bytes(
    image: np.ndarray,
    channel_type: np.int8,
) -> bytes:
    """
    Convert a NumPy image into a bytes representation.
    image: H x W x C (for color) or H x W (for grayscale)
    """

    dtype = CHANNEL_TYPE_TO_DTYPE.get(channel_type)

    if dtype is None:
        raise ValueError(f"Unsupported channel type: {channel_type}")

    # check if the image is the right type
    if (image.dtype) != dtype:
        raise ValueError(
            f"Image type {image.dtype} does not match channel type {channel_type}"
        )

    packed_data = image.tobytes()
    return packed_data


def unpack_image_from_bytes(
    data: bytes,
    height: int,
    width: int,
    num_channels: int,
    channel_type: int,
) -> np.ndarray:
    """
    Convert bytes back into a NumPy image array.
    Assumes row-major format and no padding (i.e., tightly packed).

    Arguments:
        data:          Raw byte buffer (e.g., from LCM message)
        height:        Image height
        width:         Image width
        num_channels:  Number of channels (e.g., 1=grayscale, 3=RGB)
        channel_type:  Enum value corresponding to the dtype

    Returns:
        np.ndarray with shape (H, W, C) or (H, W) if C=1
    """
    dtype = CHANNEL_TYPE_TO_DTYPE.get(channel_type)
    if dtype is None:
        raise ValueError(f"Unsupported channel type: {channel_type}")

    expected_size = height * width * num_channels * np.dtype(dtype).itemsize
    if len(data) != expected_size:
        raise ValueError(
            f"Data size mismatch: expected {expected_size} bytes, got {len(data)}"
        )

    # decode bytes from buffer into dtype
    flat_array = np.frombuffer(data, dtype=dtype)

    if num_channels == 1:
        return flat_array.reshape((height, width))
    else:
        return flat_array.reshape((height, width, num_channels))
