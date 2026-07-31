"""Load grasp prediction folders into GraspData subclasses."""


def get_grasp_data(model: str, grasp_predictions_path: str):
    if model == "robotis_5f":
        from controller.grasping.utils.Robotis5FGraspData import Robotis5FGraspData

        return Robotis5FGraspData(grasp_predictions_path)
    raise ValueError(f"Unsupported model: {model}")
