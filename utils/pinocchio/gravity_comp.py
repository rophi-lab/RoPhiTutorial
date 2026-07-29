"""Pinocchio generalized-gravity torque for the arm joints of a fixed-base robot.

Builds a Pinocchio model from a URDF and computes the joint torques required to
hold against gravity at a given configuration, g(q) = pin.computeGeneralized-
Gravity. For the Flexiv arm + Robotis hand this gives the arm's gravity-comp
torque accounting for the mounted hand's mass at its current finger pose.

The arm/hand measurement vectors are ordered by their joint-name lists (URDF
actuated order); this class maps them to Pinocchio's internal q/v indices, so
callers never deal with Pinocchio's ordering.
"""
import numpy as np
import pinocchio as pin


class ArmGravityCompensator:
    def __init__(
        self,
        urdf_path: str,
        arm_joint_names,
        hand_joint_names=None,
        gravity=(0.0, 0.0, -9.81),
    ):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.model.gravity.linear = np.asarray(gravity, dtype=float)

        self._arm_names = list(arm_joint_names)
        self._hand_names = list(hand_joint_names) if hand_joint_names else []

        self._arm_qidx = [self._q_index(n) for n in self._arm_names]
        self._hand_qidx = [self._q_index(n) for n in self._hand_names]
        self._arm_vidx = [self._v_index(n) for n in self._arm_names]
        self._nq = self.model.nq

    def _joint_id(self, name: str) -> int:
        jid = self.model.getJointId(name)
        # Pinocchio returns njoints (an invalid id) for an unknown joint.
        if jid >= self.model.njoints:
            raise ValueError(f"Joint '{name}' not found in URDF model.")
        return jid

    def _q_index(self, name: str) -> int:
        return self.model.idx_qs[self._joint_id(name)]

    def _v_index(self, name: str) -> int:
        return self.model.idx_vs[self._joint_id(name)]

    def arm_gravity(self, arm_q, hand_q=None) -> np.ndarray:
        """Gravity-comp torque (Nm) for the arm joints, in arm_joint_names order.

        @param arm_q: arm joint positions (rad), in arm_joint_names order.
        @param hand_q: optional hand joint positions (rad), in hand_joint_names
            order. Including them accounts for how the hand's mass distribution
            shifts the wrist load; omit (zeros) if you only have arm state.
        """
        q = np.zeros(self._nq)
        for i, idx in enumerate(self._arm_qidx):
            q[idx] = arm_q[i]
        if hand_q is not None and self._hand_qidx:
            for i, idx in enumerate(self._hand_qidx):
                q[idx] = hand_q[i]
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        return np.array([g[v] for v in self._arm_vidx])
