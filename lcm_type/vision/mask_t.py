"""LCM type definition for a single-object segmentation mask.

Hand-authored in the same style as rgbd_t. Carries the mask image (typically
uint8, 0/1) together with the TIMESTAMP OF THE SOURCE RGB FRAME the mask was
computed on, plus a view_id identifying which camera the mask belongs to. The
timestamp lets a downstream consumer time-match the mask against a buffered
RGBD frame (retro time alignment), so the mask producer (SAM2) and the pose
estimator (FoundationPose) can run in separate processes at different rates.
"""


from io import BytesIO
import struct


class mask_t(object):

    __slots__ = [
        "timestamp",
        "view_id",
        "height",
        "width",
        "channel_type",
        "mask_size",
        "mask_image",
    ]

    __typenames__ = [
        "double",
        "int32_t",
        "int32_t",
        "int32_t",
        "int8_t",
        "int32_t",
        "byte",
    ]

    __dimensions__ = [None, None, None, None, None, None, ["mask_size"]]

    CHANNEL_TYPE_INT8 = 0
    """ enum for channel type """
    CHANNEL_TYPE_UINT8 = 1
    CHANNEL_TYPE_INT16 = 2
    CHANNEL_TYPE_UINT16 = 3
    CHANNEL_TYPE_INT32 = 4
    CHANNEL_TYPE_UINT32 = 5
    CHANNEL_TYPE_FLOAT32 = 6
    CHANNEL_TYPE_FLOAT64 = 7

    def __init__(self):
        self.timestamp = 0.0
        """ timestamp of the source rgb frame the mask was computed on.
        LCM Type: double """
        self.view_id = 0
        """ camera/view index this mask belongs to.
        LCM Type: int32_t """
        self.height = 0
        """ LCM Type: int32_t """
        self.width = 0
        """ LCM Type: int32_t """
        self.channel_type = 0
        """ LCM Type: int8_t """
        self.mask_size = 0
        """ LCM Type: int32_t """
        self.mask_image = b""
        """ LCM Type: byte[mask_size] """

    def encode(self):
        buf = BytesIO()
        buf.write(mask_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(
            struct.pack(
                ">diiibi",
                self.timestamp,
                self.view_id,
                self.height,
                self.width,
                self.channel_type,
                self.mask_size,
            )
        )
        buf.write(bytearray(self.mask_image[: self.mask_size]))

    @staticmethod
    def decode(data: bytes):
        if hasattr(data, "read"):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != mask_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return mask_t._decode_one(buf)

    @staticmethod
    def _decode_one(buf):
        self = mask_t()
        (
            self.timestamp,
            self.view_id,
            self.height,
            self.width,
            self.channel_type,
            self.mask_size,
        ) = struct.unpack(">diiibi", buf.read(25))
        self.mask_image = buf.read(self.mask_size)
        return self

    @staticmethod
    def _get_hash_recursive(parents):
        if mask_t in parents:
            return 0
        # Fingerprint distinct from other types in this package.
        tmphash = (0x8B21C4F0A1E37D52) & 0xFFFFFFFFFFFFFFFF
        tmphash = (
            ((tmphash << 1) & 0xFFFFFFFFFFFFFFFF) + (tmphash >> 63)
        ) & 0xFFFFFFFFFFFFFFFF
        return tmphash

    _packed_fingerprint = None

    @staticmethod
    def _get_packed_fingerprint():
        if mask_t._packed_fingerprint is None:
            mask_t._packed_fingerprint = struct.pack(
                ">Q", mask_t._get_hash_recursive([])
            )
        return mask_t._packed_fingerprint

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", mask_t._get_packed_fingerprint())[0]
