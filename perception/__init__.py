from omegaconf import DictConfig

from perception.BasePerception import BasePerception


def get_perception(cfg_perception: DictConfig) -> BasePerception:
    """
    Set the perception based on the configuration provided.
    @param[in] cfg_perception: Configuration for the perception.
    """
    name = cfg_perception["name"]
    if name == "base":
        return BasePerception(cfg_perception)
    elif name == "blob_detector":
        from perception.examples.BlobDetector import BlobDetector

        return BlobDetector(cfg_perception)
    elif name == "click_and_point":
        from perception.click_and_point.ClickAndPoint import ClickAndPoint

        return ClickAndPoint(cfg_perception)
    elif name == "click_two_and_points":
        from perception.click_and_point.ClickTwoAndPoints import ClickTwoAndPoints

        return ClickTwoAndPoints(cfg_perception)
    elif name == "molmo_pointing":
        from perception.molmo.MolmoPointing import MolmoPointing

        return MolmoPointing(cfg_perception)
    elif name == "tapnet_click_and_track":
        from perception.tapnet.TapnetClickAndTrack import TapnetClickAndTrack

        return TapnetClickAndTrack(cfg_perception)
    elif name == "foundation_pose_tracking":
        from perception.foundation_pose.FoundationPoseTracking import (
            FoundationPoseTracking,
        )

        return FoundationPoseTracking(cfg_perception)
    elif name == "foundation_pose_tracking_sam2":
        from perception.foundation_pose.FoundationPoseTrackingSAM2 import (
            FoundationPoseTrackingSAM2,
        )

        return FoundationPoseTrackingSAM2(cfg_perception)
    elif name == "foundation_pose_tracking_multiview":
        from perception.foundation_pose.FoundationPoseTrackingMultiView import (
            FoundationPoseTrackingMultiView,
        )

        return FoundationPoseTrackingMultiView(cfg_perception)
    elif name == "sam2_mask_node":
        from perception.foundation_pose.SAM2MaskNode import SAM2MaskNode

        return SAM2MaskNode(cfg_perception)
    elif name == "foundation_pose_mask_node":
        from perception.foundation_pose.FoundationPoseMaskNode import (
            FoundationPoseMaskNode,
        )

        return FoundationPoseMaskNode(cfg_perception)
    elif name == "sam_segment_and_save":
        from perception.sam.SAM import SAM

        return SAM(cfg_perception)
    elif name == "sam2_multi_point_segmentation":
        from perception.sam2.SAM2MultiPointSegmentation import (
            SAM2MultiPointSegmentation,
        )

        return SAM2MultiPointSegmentation(cfg_perception)
    elif name == "sam2_multi_point_segmentation_super_quadric":
        from perception.sam2.SAM2MultiPointSegmentationSuperQuadric import (
            SAM2MultiPointSegmentationSuperQuadric,
        )

        return SAM2MultiPointSegmentationSuperQuadric(cfg_perception)
    elif name == "apriltag_detection":
        from perception.apriltag.AprilTagDetector import AprilTagDetector

        return AprilTagDetector(cfg_perception)
    else:
        raise ValueError(f"Unknown controller name: {name}.")
