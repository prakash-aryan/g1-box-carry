#!/usr/bin/env python3
"""Pick-and-place of a box by a Unitree G1 humanoid in Isaac Lab.

The environment is based on the Isaac Lab G1 loco-manipulation task:
  * lower body: pretrained "Agile" locomotion policy, commanded with [vx, vy, yaw rate, hip height];
  * upper body: Pink inverse kinematics to scripted wrist poses;
  * hands: Inspire five-finger hands in place of Dex3-1 (see g1_hands.py).

The box is grasped between the palms and held by friction; it is not attached to the robot. Task sequencing is
scripted and uses ground-truth simulator state (robot pose, box pose, scene geometry): A* path planning on an
occupancy grid, straight-line walking, alignment at the table, grasp, repositioning and re-grasp, lift,
transport, placement on the second table, and release.

Usage (from the repository root):
    python scripts/carry_box.py --scene lab
    python scripts/carry_box.py --scene warehouse
"""
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, HERE)

from g1_box_carry import nav_planning as nav
from g1_box_carry.scene_utils import collision_off_layer, occupancy_grid, rigid_prop_layer

WAREHOUSE = "physical_ai_simready_warehouse_01.usd"
BOX = "Props/general/sm_cubebox_b04_brown_01/sm_cubebox_b04_brown_01_physics.usd"

# Stage layouts in the world frame: start pose and stand poses at the pick and drop tables, as (x, y, yaw) in
# metres and degrees.
LAYOUTS = {
    "warehouse": dict(start=(-38.6, -55.0, -90.0), pick=(-38.6, -59.45, -90.0), drop=(-38.6, -47.55, 90.0)),
    "lab":       dict(start=(0.0, 4.5, -90.0),     pick=(0.0, 0.0, -90.0),      drop=(0.0, 12.0, 90.0)),
}
# Grasp geometry in the gravity-aligned body frame (metres). At wrist roll 0 the Inspire palms face each other
# with the fingers pointing forward.
HALF = 0.13           # half width of the box
GRIP_BACK = 0.10      # wrist offset behind the box centre (palm centred on the box side)
GRIP_YAW = 10.0       # outward wrist yaw (deg), compensating the inward angle of the fingers
PALM_IN = 0.015       # palm surface offset inboard of the wrist axis
SQUEEZE = 0.02        # penetration of the palm targets into the box sides
# Finger curl and thumb bend during the grasp, as fractions of the joint ranges. The thumb contact above the
# palm resists rotation of the box about the axis through both palms.
GRIP_CURL, GRIP_THUMB = 0.2, 0.1
CARRY = (0.20, 0.19)  # wrist position (forward, up) during transport
PULL_TO = 0.44        # box-centre distance to which the box is slid before re-grasping
PLACE_MIN_IN = 0.05   # minimum distance of the box's near face past the drop table edge
WRIST_REACH = 0.42    # maximum forward wrist target during placement
STAND_DIST = 0.41     # box-centre distance ahead of the pelvis at the tables
TABLE_TOP = 0.694     # table surface height
# The packing table (2.4 m x 0.8 m) is centred 0.55 m ahead of a stand pose, so its near edge is 0.15 m ahead.
# The box starts at that edge, since the table frame limits how close the robot can stand.
BOX_AHEAD = 0.15 + HALF + 0.015
TURN_RATE = 1.3       # yaw-rate command (rad/s); rates below ~1.1 rad/s are not tracked at standstill
# The arm IK also controls the waist, so a wrist target the arms cannot reach bends the waist instead (up to its
# 30 deg roll limit, leaning the torso sideways). These targets are reachable with the waist upright.
PARK = (0.14, 0.20, 0.12)  # wrist position (forward, left, up) while approaching and leaving a table
RELEASE_OUT = 0.08          # outward wrist motion after releasing the box
LEVEL_GAIN = 0.02           # carry: wrist pitch correction per control step, as a fraction of the box tilt

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--scene", choices=sorted(LAYOUTS), default="lab")
ap.add_argument("--assets", default=os.path.join(HERE, "assets", "simready_warehouse"),
                help="NVIDIA PhysicalAI-SimReady-Warehouse-01 directory (see scripts/fetch_assets.sh)")
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()
L = LAYOUTS[args.scene]
CACHE = os.path.join(HERE, ".cache")

import pinocchio  # noqa: F401  (imported before AppLauncher to load Isaac Lab's pinocchio build)
from isaaclab.app import AppLauncher

app = AppLauncher(headless=args.headless).app

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as mu
from isaaclab.assets import AssetBaseCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from g1_box_carry.g1_hands import inspire_targets, use_inspire_hands

# Isaac Lab 3.x and 2.x differ in the task module and name, the quaternion order, and the type of asset data.
try:
    import isaaclab_tasks.contrib.locomanip_pick_place  # noqa: F401
    TASK = "IsaacContrib-PickPlace-Locomanipulation-G1-Abs"
except ImportError:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    TASK = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
XYZW = AssetBaseCfg.InitialStateCfg().rot[3] == 1.0  # (x, y, z, w) in 3.x, (w, x, y, z) in 2.x


def T(x):
    """Return asset data as a torch tensor (Isaac Lab 3.x returns ProxyArray)."""
    return getattr(x, "torch", x)


def yaw_quat(deg):
    c, s = math.cos(math.radians(deg) / 2), math.sin(math.radians(deg) / 2)
    return (0.0, 0.0, s, c) if XYZW else (c, 0.0, 0.0, s)


def fwd(deg):
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


def table_pose(stand):
    """Pose of a packing table in front of a stand pose, long side across the stand heading."""
    x, y, yaw = stand
    c = np.array([x, y]) + 0.55 * fwd(yaw)
    return (float(c[0]), float(c[1]), -0.3), yaw_quat(yaw - 90.0), yaw - 90.0


# ---- 1. navigation map ------------------------------------------------------------
grid = None
if args.scene == "warehouse":
    t0 = time.time()
    grid = occupancy_grid(os.path.join(args.assets, WAREHOUSE))
    for stand in (L["pick"], L["drop"]):  # the tables are not part of the scene USD
        (tx, ty, _), _, tyaw = table_pose(stand)
        grid.add_rectangle((tx, ty), tyaw, 1.25, 0.45)
    print(f"[carry] occupancy map built in {time.time() - t0:.0f} s", flush=True)


def plan_path(a_xy, b_xy):
    """Smoothed A* path between two points; a straight line when no map is loaded."""
    if grid is None:
        return np.linspace(a_xy, b_xy, max(int(np.hypot(*(np.array(b_xy) - a_xy)) / 0.1), 2))
    ca = grid.nearest_free(*a_xy, 0.3) or grid.idx(*a_xy)
    cb = grid.nearest_free(*b_xy, 0.3) or grid.idx(*b_xy)
    p = grid.astar(ca, cb, min_clear=0.3)
    if p is None:
        raise RuntimeError(f"no path from {a_xy} to {b_xy}")
    return nav.smooth([tuple(a_xy)] + p + [tuple(b_xy)])


# ---- 2. environment ---------------------------------------------------------------
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
use_inspire_hands(cfg, os.path.join(CACHE, "robots"))
# Isaac Lab 3.x lowers the IK task gain from 0.5 to 0.075 and raises the damping from 10 to 75. The grasp
# relies on the IK driving the palms towards targets inside the box; with the 3.x values the squeeze force
# drops by a factor of 3-5 and the box slips. The 2.x values are restored.
for t in cfg.actions.upper_body_ik.controller.variable_input_tasks:
    if hasattr(t, "class_type"):  # task configurations (Isaac Lab 3.x)
        if hasattr(t, "frame"):
            t.orientation_cost, t.lm_damping, t.gain = 2.0, 10.0, 0.5
        elif hasattr(t, "controlled_frames"):
            t.cost, t.lm_damping, t.gain = 0.5, 1.0, 0.3
# Unused teleoperation components of the Isaac Lab 3.x task: the head camera and its observation, and hand
# contact sensors for XR haptics (defined on Dex3 links).
for name in ("robot_pov_cam", "left_hand_contact", "right_hand_contact"):
    if hasattr(cfg.scene, name):
        setattr(cfg.scene, name, None)
if hasattr(cfg.observations.policy, "robot_pov_cam"):
    cfg.observations.policy.robot_pov_cam = None
if hasattr(cfg, "haptic_feedback"):
    cfg.haptic_feedback = None
if hasattr(cfg, "image_obs_list"):
    cfg.image_obs_list = []
cfg.scene.env_spacing = 0.0
if args.scene == "warehouse":
    # The warehouse floor tiles use colliders that PhysX cannot cook for GPU simulation. They are disabled;
    # the robot walks on the Isaac Lab ground plane at the same height.
    wh_usd, _ = collision_off_layer(os.path.join(args.assets, WAREHOUSE), os.path.join(CACHE, "scenes"), ("a01_floor",))
    cfg.scene.warehouse = AssetBaseCfg(prim_path="/World/Warehouse", spawn=sim_utils.UsdFileCfg(usd_path=wh_usd))
    cfg.scene.light = None  # the warehouse provides its own lighting
# 0.5 kg box with a cardboard friction coefficient
box_usd, box_mpu = rigid_prop_layer(os.path.join(args.assets, BOX), os.path.join(CACHE, "props"), 0.5,
                                    friction=(0.9, 0.8))
cfg.scene.object.spawn = sim_utils.UsdFileCfg(usd_path=box_usd, scale=(box_mpu,) * 3)  # SimReady props are in cm
bx, by = np.array(L["pick"][:2]) + BOX_AHEAD * fwd(L["pick"][2])
cfg.scene.object.init_state.pos = (float(bx), float(by), TABLE_TOP + 0.005)
cfg.scene.object.init_state.rot = yaw_quat(L["pick"][2])
cfg.scene.packing_table.init_state.pos, cfg.scene.packing_table.init_state.rot, _ = table_pose(L["pick"])
drop_table = cfg.scene.packing_table.copy()
drop_table.prim_path = "/World/envs/env_.*/DropTable"
drop_table.init_state.pos, drop_table.init_state.rot, _ = table_pose(L["drop"])
cfg.scene.drop_table = drop_table
cfg.scene.robot.init_state.pos = (L["start"][0], L["start"][1], 0.75)
cfg.scene.robot.init_state.rot = yaw_quat(L["start"][2])
for term in ("object_dropping", "success", "object_too_far"):  # terminations that would interrupt the task
    if hasattr(cfg.terminations, term):
        setattr(cfg.terminations, term, None)
cfg.episode_length_s = 1e4
cfg.viewer.origin_type, cfg.viewer.asset_name = "asset_root", "robot"  # viewport camera follows the robot
cfg.viewer.eye, cfg.viewer.lookat = (2.6, 2.2, 1.4), (0.0, 0.0, 0.2)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
if args.scene == "warehouse":  # the ground plane provides contact only; the warehouse floor is rendered
    from pxr import UsdGeom
    UsdGeom.Imageable(env.scene.stage.GetPrimAtPath("/World/GroundPlane")).MakeInvisible()

robot, box = env.scene["robot"], env.scene["object"]
up = env.action_manager._terms["upper_body_ik"]
names = robot.body_names
P = names.index("pelvis")
dev = env.device
dt = env.step_dt
jl = T(robot.data.soft_joint_pos_limits)[0]
hid = [robot.joint_names.index(n) for n in up._hand_joint_names]
h_lo, h_hi = jl[hid, 0], jl[hid, 1]


# ---- 3. state and actions ---------------------------------------------------------
def pelvis():
    return T(robot.data.body_pos_w)[0, P], T(robot.data.body_quat_w)[0, P]


def pelvis_yaw():
    q = pelvis()[1].tolist()
    x, y, z, w = q if XYZW else (q[1], q[2], q[3], q[0])
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def body_frame():
    """Pelvis position with heading-only orientation. Hand targets and object positions are expressed in this
    frame so that pelvis pitch does not displace the targets."""
    return pelvis()[0], torch.tensor(yaw_quat(pelvis_yaw()), device=dev, dtype=torch.float32)


def to_body(p_world):
    pp, q = body_frame()
    return mu.quat_apply(mu.quat_inv(q), torch.as_tensor(p_world, device=dev, dtype=torch.float32) - pp).cpu().numpy()


def quat_rpy(r, p, y):
    return mu.quat_from_euler_xyz(*(torch.tensor([math.radians(v)]) for v in (r, p, y)))[0].to(dev)


def box_centre():
    return T(box.data.root_pos_w)[0].cpu().numpy() + np.array([0, 0, HALF])  # the prop's origin is at its base


def grip_report(tag):
    """Print the box and measured left-wrist positions in the body frame, with the commanded wrist position."""
    w = to_body(T(robot.data.body_pos_w)[0, names.index("left_wrist_yaw_link")])
    b = to_body(box_centre())
    print(f"[carry]   {tag}: box centre {b[0]:.2f} m ahead, left wrist {w[0]:.2f} ahead / {w[1]:.2f} left / "
          f"{w[2]:.2f} up (commanded {arms['l'][0]:.2f} / {arms['l'][1]:.2f} / {arms['l'][2]:.2f})", flush=True)


def box_pitch():
    """Box tilt (deg) about the body frame's lateral axis; positive when the top leans away from the robot."""
    up_w = mu.quat_apply(T(box.data.root_quat_w)[0], torch.tensor([0.0, 0.0, 1.0], device=dev))
    u = mu.quat_apply(mu.quat_inv(body_frame()[1]), up_w).cpu().numpy()
    return math.degrees(math.atan2(u[0], u[2]))


HOME = np.array([0.2, 0.149, 0.095])  # default wrist position (forward, left, up) in the body frame
REST = dict(roll=0.0, pitch=0.0, yaw=0.0, curl=0.15, thumb=0.4)  # wrist orientation (deg) and hand closure at rest


def action(left, right, pose, lower):
    """Build the 42-D task action: world-frame wrist poses (position and quaternion, both hands), 24 hand
    joint targets, and [vx, vy, yaw rate, hip height] for the lower-body policy.

    pose: wrist roll/pitch/yaw in degrees for the left hand (mirrored for the right), and finger curl and
    thumb bend as fractions of their ranges.
    """
    pp, q = body_frame()
    lw = pp + mu.quat_apply(q, torch.tensor(left, device=dev, dtype=torch.float32))
    rw = pp + mu.quat_apply(q, torch.tensor(right, device=dev, dtype=torch.float32))
    lq = quat_rpy(pose["roll"], pose["pitch"], pose["yaw"])
    rq = quat_rpy(-pose["roll"], pose["pitch"], -pose["yaw"])
    f = inspire_targets(up._hand_joint_names, h_lo, h_hi, pose["curl"], pose["thumb"], 0.0)
    return torch.cat([lw, mu.quat_mul(q, lq), rw, mu.quat_mul(q, rq), f,
                      torch.tensor([*lower, 0.75], device=dev, dtype=torch.float32)]).unsqueeze(0)


t_sim = 0.0
log = []
arms = {"l": HOME.copy(), "pose": dict(REST)}  # commanded left-wrist target and hand pose; the right hand is mirrored
lat0 = 0.0                                     # lateral coordinate of the mirror plane


def step(a):
    global t_sim
    env.step(a)
    t_sim += dt


def phase(name):
    log.append((round(t_sim, 1), name))
    p = pelvis()[0].tolist()
    print(f"[carry] t={t_sim:5.1f}s {name}  box z={T(box.data.root_pos_w)[0, 2].item():.3f}  "
          f"robot at ({p[0]:.2f}, {p[1]:.2f}) height {p[2]:.2f} m, heading {pelvis_yaw():.0f} deg", flush=True)


def hands(l_target, secs, lower=(0.0, 0.0, 0.0), **pose):
    """Interpolate the left wrist (right mirrored) to a body-frame target with a smoothstep profile, blending
    the given pose parameters."""
    start, p0 = arms["l"].copy(), arms["pose"]
    p1 = {**p0, **pose}
    tgt = np.array(l_target)
    n = max(int(secs / dt), 1)
    for k in range(n):
        s = min(1.0, (k + 1) / max(0.7 * n, 1))
        s = s * s * (3 - 2 * s)
        c = start + (tgt - start) * s
        pz = {key: p0[key] + (p1[key] - p0[key]) * s for key in p0}
        step(action(tuple(c), (c[0], 2 * lat0 - c[1], c[2]), pz, lower))
    arms["l"], arms["pose"] = tgt, p1


def hold(secs, lower=(0.0, 0.0, 0.0)):
    hands(arms["l"], secs, lower)


LEVEL = {"on": False}


def level():
    """Keep the held box level: it slowly rotates between the palms as the robot walks; counter-rotate both wrists
    (about 1 s time constant)."""
    if LEVEL["on"]:
        arms["pose"] = {**arms["pose"],
                        "pitch": float(np.clip(arms["pose"]["pitch"] - LEVEL_GAIN * box_pitch(), -35.0, 35.0))}


def relevel(secs=0.8):
    """Rotate the wrists to cancel the box's current tilt."""
    hands(tuple(arms["l"]), secs, pitch=float(np.clip(arms["pose"]["pitch"] - box_pitch(), -35.0, 35.0)))


def walk_cmd(lower):
    level()
    c = arms["l"]
    return action(tuple(c), (c[0], 2 * lat0 - c[1], c[2]), arms["pose"], lower)


# ---- 4. walking -------------------------------------------------------------------
# The lower-body policy tracks forward and lateral velocity commands closely but does not track small yaw-rate
# commands at standstill. Turns therefore use a fixed rate, and straight segments are held with lateral
# corrections.
def yaw_err(target_deg):
    return (target_deg - pelvis_yaw() + 180.0) % 360.0 - 180.0


def turn_to(target_deg, tol=8.0, timeout=8.0):
    t_end = t_sim + timeout
    while t_sim < t_end and abs(yaw_err(target_deg)) > tol:
        step(walk_cmd((0.0, 0.0, math.copysign(TURN_RATE, yaw_err(target_deg)))))
    hold(0.3)  # settling time after the turn


def walk_segment(a, b, v=0.45, timeout=60.0, stall_exit=False):
    """Walk along the segment a->b with forward velocity and lateral correction.

    stall_exit: also return when progress stops, e.g. on contact with a table.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = (b - a) / max(np.linalg.norm(b - a), 1e-6)
    n_left = np.array([-u[1], u[0]])
    heading = math.degrees(math.atan2(u[1], u[0]))
    if abs(yaw_err(heading)) > 10.0:
        turn_to(heading)
    t_end, hist = t_sim + timeout, []
    while t_sim < t_end:
        p = pelvis()[0].cpu().numpy()[:2]
        along = float(np.dot(b - p, u))
        if along < 0.08:
            return True
        hist.append((t_sim, along))
        if stall_exit and hist[-1][0] - hist[0][0] > 1.5:
            while hist[-1][0] - hist[0][0] > 1.5:
                hist.pop(0)
            if hist[0][1] - along < 0.02:
                return True
        if abs(yaw_err(heading)) > 20.0:
            turn_to(heading)
            continue
        # look-ahead point on the segment
        e = to_body([*(p + u * min(along, 0.6) + np.dot(a - p, n_left) * n_left), 0.0])
        step(walk_cmd((float(np.clip(1.2 * along, 0.12, v)), float(np.clip(1.5 * e[1], -0.2, 0.2)), 0.0)))
    return False


def corners(path):
    """Reduce a path to the waypoints joined by straight, collision-free segments."""
    if grid is None or len(path) <= 2:
        return [path[0], path[-1]]
    pts, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            seg = np.linspace(path[i], path[j], max(int(np.hypot(*(path[j] - path[i])) / 0.05), 2))
            cl = [grid.clearance[grid.idx(*q)] for q in seg if grid.inside(*grid.idx(*q))]
            if cl and min(cl) >= 0.3:
                break
            j -= 1
        pts.append(path[j])
        i = j
    return pts


def walk_path(path, v_max, timeout=90.0):
    pts = corners(np.asarray(path))
    t_end = t_sim + timeout
    for a, b in zip(pts[:-1], pts[1:]):
        if not walk_segment(a, b, v_max, timeout=max(t_end - t_sim, 1.0)):
            return False
    return True


def square_up(stand, tol=0.02, timeout=14.0):
    """Align with the stand heading, then correct the position with forward and lateral steps."""
    x, y, yaw = stand
    if abs(yaw_err(yaw)) > 6.0:
        turn_to(yaw, tol=6.0)
    t_end, good = t_sim + timeout, 0
    while t_sim < t_end:
        e = to_body([x, y, pelvis()[0][2].item()])
        # minimum command magnitude of 0.12 m/s; smaller commands are not tracked
        vx = 0.0 if abs(e[0]) < tol else math.copysign(max(min(abs(1.5 * e[0]), 0.2), 0.12), e[0])
        vy = 0.0 if abs(e[1]) < tol else math.copysign(max(min(abs(1.5 * e[1]), 0.2), 0.12), e[1])
        good = good + 1 if (vx == 0.0 and vy == 0.0) else 0
        step(walk_cmd((vx, vy, 0.0)))
        if good * dt > 0.4:
            break
    if abs(yaw_err(yaw)) > 6.0:
        turn_to(yaw, tol=6.0)


def approach(stand, short=0.05, v=0.3):
    """Walk along the table axis until the stand pose is reached or the table blocks further progress."""
    s, f = np.array(stand[:2]), fwd(stand[2])
    walk_segment(s - 1.0 * f, s - short * f, v=v, timeout=12.0, stall_exit=True)


def pre_stand(stand, dist=1.0):
    return tuple(np.array(stand[:2]) - dist * fwd(stand[2]))


# ---- 5. task sequence -------------------------------------------------------------
with torch.inference_mode():
    phase("settle")
    hold(1.5)

    phase("walk to packing table")
    walk_path(plan_path(pelvis()[0].cpu().numpy()[:2], pre_stand(L["pick"])), v_max=0.5)
    phase("square up")
    hands(PARK, 0.8)  # retract the hands for the approach
    stand_pick = (*(box_centre()[:2] - STAND_DIST * fwd(L["pick"][2])), L["pick"][2])
    approach(stand_pick)
    square_up(stand_pick, timeout=4.0)  # the table frame stops the robot short; the grasp uses the measured box pose
    b = to_body(box_centre())
    if b[0] > 0.65 or abs(b[1]) > 0.06:  # box outside the reachable region
        square_up(stand_pick, timeout=10.0)

    phase("reach")
    b = to_body(box_centre())
    lat0 = float(b[1])
    wf, wz, W = b[0] - GRIP_BACK, b[2], HALF + PALM_IN
    hands((0.25, lat0 + W + 0.12, wz + 0.03), 1.2, yaw=GRIP_YAW, thumb=0.0)
    hands((wf, lat0 + W + 0.08, wz), 1.2)
    phase("grip")
    hands((wf, lat0 + W - SQUEEZE, wz), 0.8)
    hands((wf, lat0 + W - SQUEEZE, wz), 0.6, curl=GRIP_CURL, thumb=GRIP_THUMB)
    # If the box centre lies beyond the reach required for a centred grasp, the grasp is behind the centre of
    # mass and the box pitches forward. The box is then slid towards the robot, released and grasped again at
    # its centre.
    b = to_body(box_centre())
    if b[0] > PULL_TO + 0.02:
        phase("pull closer")
        hands((arms["l"][0] - (b[0] - PULL_TO), arms["l"][1], wz), 1.2)
        hands(tuple(arms["l"]), 0.4, curl=0.0, thumb=0.0)
        hands((arms["l"][0], arms["l"][1] + 0.05, wz), 0.5)
        b = to_body(box_centre())
        lat0 = float(b[1])
        wf, wz = b[0] - GRIP_BACK, b[2]
        hands((wf, lat0 + W + 0.05, wz), 0.8)
        hands((wf, lat0 + W - SQUEEZE, wz), 0.6)
        hands((wf, lat0 + W - SQUEEZE, wz), 0.5, curl=GRIP_CURL, thumb=GRIP_THUMB)
    phase("lift")
    hands((wf - 0.03, lat0 + W - SQUEEZE, wz + 0.12), 1.2)
    hands((CARRY[0], lat0 + W - SQUEEZE, CARRY[1]), 1.5)
    print(f"[carry]   box pitch after lift: {box_pitch():+.1f} deg", flush=True)
    relevel()
    LEVEL["on"] = True
    grip_report("lifted")
    phase("step back")
    hold(1.6, lower=(-0.3, 0.0, 0.0))

    phase("carry")
    walk_path(plan_path(pelvis()[0].cpu().numpy()[:2], pre_stand(L["drop"])), v_max=0.45)
    phase("square up at drop table")
    drop_c = np.array(L["drop"][:2]) + BOX_AHEAD * fwd(L["drop"][2])
    stand_drop = (*(drop_c - STAND_DIST * fwd(L["drop"][2])), L["drop"][2])
    approach(stand_drop)
    square_up(stand_drop, timeout=5.0)

    phase("place")
    relevel()
    LEVEL["on"] = False
    print(f"[carry]   box pitch before placing: {box_pitch():+.1f} deg", flush=True)
    grip_report("at the drop table")
    # Closed-loop placement on the measured box position: the box is moved forward until its near face is
    # PLACE_MIN_IN past the table edge (stepping forward when the wrist target would exceed WRIST_REACH), then
    # lowered onto the table.
    edge_w = np.array([*(np.array(L["drop"][:2]) + 0.15 * fwd(L["drop"][2])), 0.0])

    def past_edge():  # distance of the box's near face past the table's near edge
        return float(to_body(box_centre())[0] - HALF - to_body(edge_w)[0])

    clear = TABLE_TOP + 0.03 - T(box.data.root_pos_w)[0, 2].item()  # raise the box bottom to 3 cm above the tabletop
    if clear > 0:
        hands((arms["l"][0], arms["l"][1], arms["l"][2] + clear), 0.6)
    for _ in range(8):
        need = PLACE_MIN_IN + 0.02 - past_edge()
        if need <= 0.0:
            break
        if arms["l"][0] + need > WRIST_REACH:
            hold(0.8, lower=(0.15, 0.0, 0.0))
        else:
            hands((arms["l"][0] + need, arms["l"][1], arms["l"][2]), 0.8)
    print(f"[carry]   box near face {past_edge():+.2f} m past the table edge", flush=True)
    down = T(box.data.root_pos_w)[0, 2].item() - TABLE_TOP  # the prop's origin is at its base
    hands((arms["l"][0], arms["l"][1], arms["l"][2] - down - 0.01), 1.0)
    hold(0.4)  # settling time before release
    phase("release")
    # Release: open the hands, move them outward with the wrists yawed outward so the fingers clear the box,
    # retract past the box's near face, step back, and return the arms to the rest pose.
    hands(tuple(arms["l"]), 0.5, curl=0.0, thumb=0.0)
    hands((arms["l"][0], arms["l"][1] + RELEASE_OUT, arms["l"][2]), 1.0, yaw=30.0)
    b = to_body(box_centre())
    print(f"[carry]   released: box centre {b[0]:.2f} m ahead, z={box_centre()[2]:.3f}", flush=True)
    hands(PARK, 1.0, yaw=0.0)  # clear of the box's near face, in a pose the arms reach without the waist
    phase("step back")
    hold(1.5, lower=(-0.25, 0.0, 0.0))
    lat0 = 0.0
    hands(tuple(HOME), 1.2, **REST)
    hold(1.0)
    phase("done")

    bc = box_centre()
    on_table = abs(bc[2] - TABLE_TOP - HALF) < 0.03 and past_edge() > 0.0
    result = {"phases": log, "box_final": [round(float(v), 3) for v in bc], "box_on_drop_table": bool(on_table)}
    print(json.dumps(result), flush=True)
    if not args.headless:  # keep rendering until the window is closed
        while app.is_running():
            env.sim.render()
os._exit(0)  # app.close() may block during Kit shutdown
