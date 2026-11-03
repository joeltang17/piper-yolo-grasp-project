"""Observed-scene checks for a simplified hand and carried object, not the arm."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from piper_app.calibration.transforms import translation_rotation_delta


def transform_points(points, rotation, translation=0):
    # Nx3 transforms are too small per row to benefit from threaded BLAS.
    # Explicit einsum avoids its dispatch/thread overhead without subsampling.
    return np.einsum('ij,jk->ik', points, rotation, optimize=False) + translation


def hand_hits(points, pose, opening, config):
    # Business TCP = middle of pads; approach=P.Z, closing=P.Y.
    p = transform_points(points - pose[:3, 3], pose[:3, :3])
    x, y, z = p[:, 2], p[:, 1], -p[:, 0]
    ec = config["execution"]; margin = ec["geometry_margin_m"]
    half = config["pad_length_m"] / 2
    finger = ((x >= -half-margin) & (x <= half+margin)
              & (np.abs(y) >= opening/2-margin) & (np.abs(y) <= opening/2+.01+margin)
              & (np.abs(z) <= .01+margin))
    palm = ((x >= -half-.07-margin) & (x <= -half+margin)
            & (np.abs(y) <= max(.055, opening/2+.015)+margin) & (np.abs(z) <= .025+margin))
    return finger | palm


def interpolate(start, end):
    d, a = translation_rotation_delta(start, end)
    count = max(1, int(np.ceil(d/.01)), int(np.ceil(a/5)))
    rot = Slerp([0, 1], Rotation.from_matrix(np.array([start[:3, :3], end[:3, :3]])))
    for f in np.linspace(0, 1, count+1)[1:]:
        t = np.eye(4); t[:3, :3] = rot(f).as_matrix()
        t[:3, 3] = (1-f)*start[:3, 3] + f*end[:3, 3]
        yield t


def prepare_scene(start, obj, context, plane, config):
    # Voxel sampling bounds cost while retaining all parts of the observed scene.
    _, indices = np.unique(np.floor(context/.003).astype(np.int32), axis=0, return_index=True)
    points = context[indices]
    if np.any(hand_hits(obj, start, config["max_opening_m"], config)):
        raise RuntimeError("Selected object overlaps the initial hand envelope.")
    # Exclude the hand's own observed returns at its measured starting pose.
    points = points[~hand_hits(points, start, config["max_opening_m"], config)]
    # Use the existing geometry tolerance for observed target-surface returns.
    # A stricter fixed 3mm cut mislabeled 3–4mm mask/depth edge returns as a
    # stationary neighbour that the very same object would hit on lifting.
    surface_tolerance=min(.004,max(.003,float(config['execution']['geometry_margin_m'])))
    object_distance, _ = cKDTree(obj).query(points,distance_upper_bound=surface_tolerance+1e-12)
    other = points[object_distance > surface_tolerance]
    h = other[:, 2] - np.c_[other[:, :2], np.ones(len(other))] @ plane
    air_release = config["execution"].get("placement_mode", "taught_release") == "taught_release"
    # An air drop has no intended support contact: retain observed table points
    # as obstacles, but do not extend the pickup plane infinitely under a bin.
    obstacles = other if air_release else other[h > .008]
    tree = cKDTree(obstacles) if len(obstacles) else None
    above_support=other[h>.006]
    lift_tree=cKDTree(above_support) if len(above_support) else None
    return points, other, tree, air_release, cKDTree(points), cKDTree(other), lift_tree


def hand_radius(opening, config):
    m=config['execution']['geometry_margin_m']
    return 1.01*float(np.linalg.norm([config['pad_length_m']/2+.07+m,
                    max(.055,opening/2+.015)+m,.025+m]))+.001


def nearby(points, tree, centre, radius):
    return points[tree.query_ball_point(centre,radius)]


def payload_hits(tree, points):
    if tree is None:
        return False
    distances,indices=tree.query(points,distance_upper_bound=.006)
    # Count independent occupied scene voxels, not repeated matches to the
    # same noisy edge return from many densely sampled target points.
    return len(np.unique(indices[distances<.006]))>3


def check_open_path(poses, points, camera_offset, opening, config, label, *, tree=None):
    tree=cKDTree(points) if tree is None else tree
    radius=hand_radius(opening,config)
    for pose in poses:
        local=nearby(points,tree,pose[:3,3],radius)
        if np.count_nonzero(hand_hits(local,pose,opening,config)) > 5:
            raise RuntimeError(f"Observed scene intersects the hand envelope on {label}.")
        camera = pose @ camera_offset
        local=nearby(points,tree,camera[:3,3],.065)
        c = transform_points(local-camera[:3,3],camera[:3,:3])
        body = (np.abs(c[:,0]) < .05) & (np.abs(c[:,1]) < .02) & (c[:,2] > -.035) & (c[:,2] < -.003)
        if np.count_nonzero(body) > 5:
            raise RuntimeError(f"Observed scene intersects the camera envelope on {label}.")


def check_gripper_paths(start, matrices, base_camera, row, obj, context, plane, config, *, segment_paths=None, prepared=None):
    if segment_paths is not None and len(segment_paths) != len(matrices):
        raise ValueError("Every stage must have a corresponding path description.")
    points, other, tree, air_release, points_tree, other_tree, lift_tree = prepare_scene(start,obj,context,plane,config) if prepared is None else prepared
    grasp = dict(matrices)["grasp"]
    object_local = transform_points(obj - grasp[:3, 3], grasp[:3, :3])
    initial_low=float(np.percentile(obj[:,2]-(obj[:,0]*plane[0]+obj[:,1]*plane[1]+plane[2]),1))
    camera_offset = np.linalg.inv(start) @ base_camera
    place = dict(matrices)['place']
    released = transform_points(object_local, place[:3,:3].T, place[:3,3])
    released_context=np.concatenate((other,released))
    released_tree=cKDTree(released_context)
    previous = start
    for stage_index, (name, target) in enumerate(matrices):
        carrying = name in {"lift", "above_place", "place"}
        opening = row[1] if carrying else config["max_opening_m"]
        # After release the placed object is also an obstacle during withdrawal.
        if name in {"clear", "clear_up", "observe"}:
            hand_context, hand_tree = released_context, released_tree
        else:
            hand_context, hand_tree = (other,other_tree) if carrying else (points,points_tree)
        radius=hand_radius(opening,config)
        supplied = None if segment_paths is None else segment_paths[stage_index]
        if supplied is not None:
            if not len(supplied):
                raise ValueError('Joint path samples cannot be empty.')
            d,a=translation_rotation_delta(supplied[-1],target)
            if d>.002 or a>1.:
                raise ValueError("Joint path FK endpoint does not meet the commanded TCP pose tolerance.")
        # Joint transfers curve through FK poses; a Cartesian chord checks a
        # different route and can both miss obstacles and invent collisions.
        for pose in interpolate(previous, target) if supplied is None else supplied:
            local=nearby(hand_context,hand_tree,pose[:3,3],radius)
            if np.count_nonzero(hand_hits(local, pose, opening, config)) > 5:
                raise RuntimeError(f"Observed scene intersects the hand envelope on {name}.")
            # Camera body envelope in its calibrated optical frame. Geometry is
            # conservative nominal D435 dimensions, not a full mounting model.
            camera = pose @ camera_offset
            local=nearby(hand_context,hand_tree,camera[:3,3],.065)
            c = transform_points(local - camera[:3, 3], camera[:3, :3])
            body = (np.abs(c[:, 0]) < .05) & (np.abs(c[:, 1]) < .02) & (c[:, 2] > -.035) & (c[:, 2] < -.003)
            if np.count_nonzero(body) > 5:
                raise RuntimeError(f"Observed scene intersects the camera envelope on {name}.")
            if carrying:
                moved = transform_points(object_local, pose[:3, :3].T, pose[:3, 3])
                height = moved[:, 2] - (moved[:,0]*plane[0]+moved[:,1]*plane[1]+plane[2])
                if name=='lift' and np.percentile(height,1)<initial_low-.002:
                    raise RuntimeError('Lift moves the carried object farther into the pickup support.')
                if not air_release and np.percentile(height, 1) < -.006:
                    raise RuntimeError(f"Carried object penetrates the support plane on {name}.")
                # The pickup surface is intended initial contact, not a new
                # obstacle during vertical separation. Keep elevated objects;
                # all observed floor points remain obstacles during transport.
                payload_tree=lift_tree if name=='lift' and air_release else tree
                if payload_hits(payload_tree,moved[::8]):
                    raise RuntimeError(f"Carried object intersects another observed object on {name}.")
        if name=='lift':
            moved=transform_points(object_local,target[:3,:3].T,target[:3,3])
            height=moved[:,2]-(moved[:,0]*plane[0]+moved[:,1]*plane[1]+plane[2])
            if np.percentile(height,1)<.006:
                raise RuntimeError('Lift does not clear the measured pickup support.')
        previous = supplied[-1] if supplied is not None else target
