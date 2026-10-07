"""Scene preparation: USD override layers and an occupancy map.

The override layers are composed over the original assets, which are not modified, and are written to a cache
directory. pxr is imported inside the functions so that the module can be imported before Isaac Sim starts.
"""
import os

import numpy as np

from g1_box_carry import nav_planning as nav


def collision_off_layer(scene_path, cache_dir, patterns):
    """Write a layer over a scene that disables every collider whose path contains one of `patterns`. Returns
    (layer path, number of colliders disabled)."""
    from pxr import Sdf, Usd, UsdPhysics

    src = Usd.Stage.Open(scene_path)
    targets = [p.GetPath() for p in Usd.PrimRange(src.GetDefaultPrim(), Usd.TraverseInstanceProxies())
               if p.HasAPI(UsdPhysics.CollisionAPI) and any(k in str(p.GetPath()) for k in patterns)]
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, os.path.splitext(os.path.basename(scene_path))[0] + "_nav.usda")
    layer = Sdf.Layer.CreateNew(out) if not os.path.exists(out) else Sdf.Layer.FindOrOpen(out)
    layer.Clear()
    layer.subLayerPaths.append(os.path.abspath(scene_path))
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(layer)
    if src.GetDefaultPrim():
        layer.defaultPrim = src.GetDefaultPrim().GetName()
    for path in targets:
        UsdPhysics.CollisionAPI(stage.OverridePrim(path)).CreateCollisionEnabledAttr(False)
    layer.Save()
    return out, len(targets)


def rigid_prop_layer(prop_path, cache_dir, mass_kg, friction=None):
    """Write a layer that makes a SimReady prop a dynamic rigid body.

    SimReady assets define colliders but no RigidBodyAPI, which Isaac Lab's RigidObject requires on the root
    prim. friction: (static, dynamic) coefficients bound as a physics material; the props' own physics
    materials are referenced from an omniverse:// server and are not resolved offline, in which case PhysX
    uses its default of 0.5.

    Returns (layer path, metersPerUnit of the prop). SimReady props are authored in centimetres and are
    spawned with scale = metersPerUnit.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    src = Usd.Stage.Open(prop_path)
    mpu = UsdGeom.GetStageMetersPerUnit(src) or 1.0
    root = src.GetDefaultPrim()
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"_mu{friction[0]:g}" if friction else ""
    out = os.path.join(cache_dir, os.path.splitext(os.path.basename(prop_path))[0] + f"_rigid_{mass_kg:g}kg{tag}.usda")
    layer = Sdf.Layer.CreateNew(out) if not os.path.exists(out) else Sdf.Layer.FindOrOpen(out)
    layer.Clear()
    layer.subLayerPaths.append(os.path.abspath(prop_path))
    layer.defaultPrim = root.GetName()
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(layer)
    prim = stage.OverridePrim(root.GetPath())
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass_kg))
    if friction:
        mat = UsdShade.Material.Define(stage, root.GetPath().AppendChild("GripMaterial"))
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(float(friction[0]))
        pm.CreateDynamicFrictionAttr(float(friction[1]))
        pm.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    layer.Save()
    return out, mpu


def face_boxes(stage, z_band=None, mpu=None):
    """Per-face axis-aligned boxes of all visible meshes, in metres, as (lo, hi) arrays of shape (N, 3).
    Per-face boxes are used because mesh-level bounding boxes over-approximate non-convex geometry.

    z_band: optional (zmin, zmax); faces entirely outside it are discarded.
    """
    from pxr import Usd, UsdGeom

    mpu = mpu or UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    t = Usd.TimeCode.Default()
    xf_cache = UsdGeom.XformCache(t)
    los, his = [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(t) == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get(t)
        counts = mesh.GetFaceVertexCountsAttr().Get(t)
        idx = mesh.GetFaceVertexIndicesAttr().Get(t)
        if not pts or not counts or not idx:
            continue
        P = np.asarray(pts, dtype=np.float64)
        M = np.asarray(xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        W = (P @ M[:3, :3] + M[3, :3]) * mpu  # USD uses row vectors
        if z_band is not None and (W[:, 2].max() < z_band[0] or W[:, 2].min() > z_band[1]):
            continue
        counts = np.asarray(counts, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        if len(counts) == 0 or counts.sum() != len(idx) or idx.max() >= len(W):
            continue
        V = W[idx]
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        lo = np.minimum.reduceat(V, starts, axis=0)
        hi = np.maximum.reduceat(V, starts, axis=0)
        if z_band is not None:
            keep = (hi[:, 2] >= z_band[0]) & (lo[:, 2] <= z_band[1])
            lo, hi = lo[keep], hi[keep]
        los.append(lo)
        his.append(hi)
    if not los:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(los), np.concatenate(his)


def occupancy_grid(scene_path):
    """nav_planning.Grid over the whole scene, with geometry between 5 cm and 90 cm above the floor marked
    occupied. The floor height is the top of the largest thin slab nearest to z = 0."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(scene_path)
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("only Z-up scenes are supported")
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    boxes = []
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not r.IsEmpty():
                lo, hi = [v * mpu for v in r.GetMin()], [v * mpu for v in r.GetMax()]
                if max(abs(v) for v in lo + hi) < 1e5:
                    boxes.append((lo, hi))
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    window = (lo[0], lo[1], hi[0], hi[1])
    area = (hi[0] - lo[0]) * (hi[1] - lo[1])
    slabs = [b for b in boxes if (b[1][0] - b[0][0]) * (b[1][1] - b[0][1]) > 0.3 * area and b[1][2] - b[0][2] < 0.5]
    floor = min((b[1][2] for b in slabs), key=abs) if slabs else 0.0
    faces = face_boxes(stage, z_band=(floor - 0.1, floor + 2.8), mpu=mpu)
    return nav.Grid(faces, window, floor)
