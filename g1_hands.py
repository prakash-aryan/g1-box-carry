"""Replace the Dex3-1 hands of the Isaac Lab G1 loco-manipulation task with Inspire hands.

The task uses the G1 with three-finger Dex3-1 hands. Isaac Lab also provides the 29-DoF G1 with Inspire RH56
five-finger hands. The lower-body policy observes only body joints (arms, waist, legs), so the hands can be
exchanged; the Pink IK controller requires a URDF whose joints match the simulated articulation.

The Inspire USD defines its articulation root on "root_joint", a fixed joint to the world, which prevents a
floating-base articulation. floating_base_layer() deactivates root_joint and places the articulation root on
the pelvis.

Call use_inspire_hands(cfg, cache_dir) after parse_env_cfg and before gym.make.
"""
import os

# Drive and mimic joints of both hands, in the order used by the Isaac Lab Inspire task.
INSPIRE_HAND_JOINTS = [
    "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint", "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint", "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint", "L_middle_intermediate_joint", "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint", "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint", "R_middle_intermediate_joint", "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint", "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint", "R_thumb_intermediate_joint",
    "L_thumb_distal_joint", "R_thumb_distal_joint",
]

# Link-name prefix in the task's Dex3 kinematics URDF, and in the URDF exported from the Inspire USD (the
# Isaac Sim 6.x exporter omits the robot-name prefix).
_DEX3_PREFIX = "g1_29dof_with_hand_rev_1_0_"
_INSPIRE_PREFIX = {5: "g1_29dof_rev_1_0_", 6: ""}


def floating_base_layer(usd_url, cache_dir):
    """Write an override layer that gives the Inspire G1 a floating base; return its path."""
    from pxr import Sdf, Usd, UsdPhysics

    src = Usd.Stage.Open(usd_url)
    root = src.GetDefaultPrim()
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, os.path.splitext(os.path.basename(usd_url))[0] + "_floating.usda")
    layer = Sdf.Layer.CreateNew(out) if not os.path.exists(out) else Sdf.Layer.FindOrOpen(out)
    layer.Clear()
    layer.subLayerPaths.append(usd_url)
    layer.defaultPrim = root.GetName()
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(layer)
    stage.OverridePrim(root.GetPath().AppendChild("root_joint")).SetActive(False)
    pelvis = stage.OverridePrim(root.GetPath().AppendChild("pelvis"))
    UsdPhysics.ArticulationRootAPI.Apply(pelvis)
    pelvis.AddAppliedSchema("PhysxArticulationAPI")
    layer.Save()
    return out


def use_inspire_hands(cfg, cache_dir):
    """Configure the task for the Inspire hands: robot USD, hand actuators, IK hand joints, IK frame names and
    IK URDF."""
    import isaaclab.controllers.utils as ControllerUtils
    from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

    robot = cfg.scene.robot
    robot.spawn.usd_path = floating_base_layer(G1_INSPIRE_FTP_CFG.spawn.usd_path, cache_dir)
    robot.actuators["hands"] = G1_INSPIRE_FTP_CFG.actuators["hands"]  # includes the ring and pinky joints

    ik = cfg.actions.upper_body_ik
    ik.hand_joint_names = list(INSPIRE_HAND_JOINTS)
    ctl = ik.controller
    ctl.num_hand_joints = len(INSPIRE_HAND_JOINTS)
    # IK frame names: Pink task objects in Isaac Lab 2.x, task configurations in 3.x
    isaac_lab_3 = hasattr(ctl, "usd_path")
    ren = lambda name: name.replace(_DEX3_PREFIX, _INSPIRE_PREFIX[6 if isaac_lab_3 else 5])
    for t in ctl.variable_input_tasks:
        for attr in ("frame", "base_link_frame_name"):
            if isinstance(getattr(t, attr, None), str):
                setattr(t, attr, ren(getattr(t, attr)))
        if getattr(t, "controlled_frames", None):
            t.controlled_frames = [ren(f) for f in t.controlled_frames]
    # IK URDF generated from the Inspire USD; Isaac Lab 3.x performs the conversion itself
    if isaac_lab_3:
        ctl.usd_path, ctl.urdf_output_dir = G1_INSPIRE_FTP_CFG.spawn.usd_path, os.path.join(cache_dir, "ik")
        ctl.urdf_path, ctl.mesh_path = None, None
    else:
        ctl.urdf_path, ctl.mesh_path = ControllerUtils.convert_usd_to_urdf(
            G1_INSPIRE_FTP_CFG.spawn.usd_path, os.path.join(cache_dir, "ik"), force_conversion=True)
    # observation term defined on Dex3 joint names (".*_hand.*"); not used
    cfg.observations.policy.hand_joint_state = None


# PhysxMimicJointAPI couplings in the Inspire USD (q = -gearing * q_ref): the intermediate joint of each
# finger follows its proximal joint 1:1; the thumb's intermediate and distal joints follow its proximal pitch
# joint at 1.6 and 2.4.
_MIMIC = {"index_intermediate": ("index_proximal", 1.0), "middle_intermediate": ("middle_proximal", 1.0),
          "ring_intermediate": ("ring_proximal", 1.0), "pinky_intermediate": ("pinky_proximal", 1.0),
          "thumb_intermediate": ("thumb_proximal_pitch", 1.6), "thumb_distal": ("thumb_proximal_pitch", 2.4)}


def inspire_targets(joint_names, lo, hi, curl, thumb_pitch, thumb_yaw):
    """Hand joint targets from the six actuated DoF per hand, given as fractions of their ranges: finger curl
    (all four fingers), thumb pitch and thumb yaw. Mimic joints are set according to their couplings, so the
    joint drives are consistent with the mimic constraints.

    lo, hi: joint limit tensors.
    """
    import torch

    out = lo.clone()
    idx = {n: i for i, n in enumerate(joint_names)}
    for n, i in idx.items():
        key = n[2:-len("_joint")]  # "L_index_proximal_joint" -> "index_proximal"
        frac = {"thumb_proximal_yaw": thumb_yaw, "thumb_proximal_pitch": thumb_pitch}.get(
            key, curl if key.endswith("_proximal") else None)
        if frac is not None:
            out[i] = lo[i] + frac * (hi[i] - lo[i])
    for n, i in idx.items():
        key = n[2:-len("_joint")]
        if key in _MIMIC:
            ref, ratio = _MIMIC[key]
            out[i] = torch.clamp(ratio * out[idx[f"{n[:2]}{ref}_joint"]], lo[i], hi[i])
    return out
