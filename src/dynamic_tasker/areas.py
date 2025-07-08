# areas.py
import numpy as np
from itertools import combinations

# ---------- small utilities ---------------------------------------------------
EPS = 1e-12

def normalize(v: np.ndarray) -> np.ndarray:
    """Return v / ‖v‖."""
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("zero‑length vector")
    return v / n

def interior_signs(planes, interior_pt):
    """
    Return σ_j = ±1 for every plane, where +1 means the interior point is on
    the 'inside' side ( n·x > d ).
    """
    return [1 if np.dot(p["n"], interior_pt) - p["d"] > 0 else -1
            for p in planes]


def vertex_angle(v, n_prev, n_next):
    """
    Corner angle θ at vertex v where the boundary turns from plane n_prev
    to plane n_next (both unit normals).
    """
    t1 = normalize(np.cross(n_prev, v))
    t2 = normalize(np.cross(n_next, v))
    return np.arccos(np.clip(np.dot(t1, t2), -1.0, 1.0))


def edge_contribution(v1, v2, n, d, sigma):
    """
    ∫ κ_g ds of the small‑circle arc between v1 and v2 on plane (n,d),
    with orientation sign sigma = ±1.
    """
    α = np.arccos(d)                  # angular radius of the circle
    if abs(np.cos(α)) < 1e-14:       # great circle ⇒ κ_g = 0
        return 0.0

    # project the vertices onto the plane to obtain circle centre‑angle Δψ
    u1 = v1 - d * n
    u2 = v2 - d * n
    denom = np.sin(α) ** 2
    cos_Δψ = np.clip(np.dot(u1, u2) / denom, -1.0, 1.0)
    Δψ = np.arccos(cos_Δψ)

    return sigma * np.cos(α) * Δψ
# -----------------------------------------------------------------------------

def already_used(u, used):
    return any(np.allclose(u, w, atol=EPS) for w in used)

def spherical_polygon_area(vertices,                    # [N,3]
                           planes,                      # list of {"n":..., "d":...}
                           edge_plane_indices,          # [N] plane index for edge i→i+1
                           interior_pt=None):
    """
    Area of the region on the unit sphere bounded by the given planes.

    Parameters
    ----------
    vertices            CCW‑ordered vertices on S² (N×3 array‑like)
    planes              [{'n': unit normal (3,), 'd': scalar}, …]
    edge_plane_indices  plane index of edge (v_i → v_{i+1})
    interior_pt         optional point known to lie inside the patch
                        (defaults to mean of vertices, then renormalised)

    Returns
    -------
    area  (float) – steradians
    """
    V = [normalize(np.asarray(v, float)) for v in vertices]
    N = len(V)

    if interior_pt is None:
        interior_pt = normalize(np.mean(V, axis=0))

    σ = interior_signs(planes, interior_pt)

    corner_sum = 0.0
    edge_sum   = 0.0

    for i in range(N):
        v_curr = V[i]
        v_next = V[(i + 1) % N]

        plane_prev = planes[edge_plane_indices[i - 1]]
        plane_next = planes[edge_plane_indices[i]]

        # --- corner term θ_i -----------------------------------------------
        corner_sum += vertex_angle(v_curr,
                                   plane_prev["n"],
                                   plane_next["n"])

        # --- edge line‑integral term ---------------------------------------
        idx = edge_plane_indices[i]
        p   = planes[idx]
        edge_sum += edge_contribution(v_curr, v_next,
                                      p["n"], p["d"], σ[idx])

    # Gauss–Bonnet on S²  →  A = 2π − Σθ_i − Σ∫κ_g ds
    return 2.0 * np.pi - corner_sum - edge_sum

def are_planes_equal(p1, p2, tol=1e-9):
    return (np.allclose(p1["n"], p2["n"], atol=tol) and
            np.isclose(p1["d"],  p2["d"], atol=tol))


def vertex_set_from_planes(planes, interior_pt=None):
    # Start first by calculating all intersections of the planes
    lines = []

    # Find the intersection of each pair of planes
    for p,q in combinations(planes, 2):
        # if np.all(p["n"] == q["n"]) and np.all(p["d"] == q["d"]):
        #     continue
        
        n1 = p["n"]
        n2 = q["n"]
        d1 = p["d"]
        d2 = q["d"]

        v = np.cross(n1, n2)
        if np.linalg.norm(v) < EPS:
            continue
            
        p0 = np.cross(d1 * n2 - d2 * n1, v)/np.linalg.norm(v)**2

        lines.append((p0, v))

    vertices = []
    # Calculate vertices as lines projected to unit magnitude
    for i, (p0, v) in enumerate(lines):
        a = v.dot(v)
        b = 2 * p0.dot(v)
        c = p0.dot(p0) - 1.0

        delta = b**2 - 4 * a * c
        if delta < 0:
            continue

        sqrt_delta = np.sqrt(delta)

        t1 = (-b + sqrt_delta) / (2 * a)
        t2 = (-b - sqrt_delta) / (2 * a)

        vertices.append(p0 + t1 * v)
        vertices.append(p0 + t2 * v)

    # Renormalize the vertices to unit length
    vertices = [normalize(v) for v in vertices]

    # Remove ones that aren't tightly bound by the planes
    vertices_tight = [v for v in vertices if np.all([np.dot(p["n"], v) - p["d"] >= -EPS for p in planes])]       
    # Order the vertices
    vertices_sorted = [vertices_tight[0]]
    planes_used = []
    edge_plane_indices = []


    # centre = normalize(np.mean(vertices_tight, axis=0))
    # choose an arbitrary in‑plane x‑axis
    centre = normalize(np.mean(vertices_tight, axis=0)) if interior_pt is None else normalize(interior_pt)
    x0 = (np.cross([1, 0, 0], centre))
    if np.linalg.norm(x0) < EPS:               # centre ≈ pole
        x0 = np.array([1, 0, 0])
    else:
        x0 = normalize(x0)
    
    y0 = np.cross(centre, x0)

    angles = [np.arctan2(np.dot(y0, v), np.dot(x0, v)) for v in vertices_tight]
    order  = np.argsort(angles)
    vertices_sorted = [vertices_tight[i] for i in order]

    # for i in range(len(vertices_tight) - 1):
    #     v = vertices_sorted[-1]
    #     # Find the plane that is closest to the vertex
    #     planes_close = [p for p in planes if np.abs(np.dot(p["n"], v) - p["d"]) <= EPS and (True if len(planes_used) == 0 else not are_planes_equal(p, planes_used[-1]))]
    #     if len(planes_close) == 0:
    #         raise ValueError("No plane found for vertex")
    #     plane = planes_close[-1]
    #     planes_used.append(plane)
    #     # edge_plane_indices.append(planes.index(plane))
    #     edge_plane_indices.append([i for i, p in enumerate(planes) if are_planes_equal(p, plane)][0])

    #     # Find the next vertex that is closest to the plane
    #     next_vertex = None
    #     next_vertex = [v for v in vertices_tight if np.abs(np.dot(plane["n"], v) - plane["d"]) <= EPS and (v != vertices_sorted[-1]).any()]
    #     if next_vertex:
    #         vertices_sorted.append(next_vertex[0])
    #     else:
    #         raise ValueError("No next vertex found for the current vertex")

    # # Add the final plane to the list
    # # This plane connects the last vertex to the first vertex
    # plane_final = [p for p in planes if np.abs(np.dot(p["n"], vertices_sorted[-1]) - p["d"]) <= EPS and np.abs(np.dot(p["n"], vertices_sorted[0]) - p["d"]) <= EPS]
    # if len(plane_final) == 0:
    #     raise ValueError("No final plane found for the last vertex")
    
    # planes_used.append(plane_final[-1])
    # edge_plane_indices.append([i for i, p in enumerate(planes) if are_planes_equal(p, plane_final[-1])][0])
    edge_plane_indices = []
    for i in range(len(vertices_sorted)):
        v_cur, v_nxt = vertices_sorted[i], vertices_sorted[(i+1) % len(vertices_sorted)]
        shared = [k for k,p in enumerate(planes)
                if abs(np.dot(p["n"], v_cur) - p["d"]) <= EPS
                and abs(np.dot(p["n"], v_nxt) - p["d"]) <= EPS]

        # pick the plane that is *not* the previous one to maintain CCW order
        if i and shared[0] == edge_plane_indices[-1]:
            edge_plane_indices.append(shared[1])
        else:
            edge_plane_indices.append(shared[0])

    return vertices_sorted, planes, edge_plane_indices