import numpy as np
import pytest

import os
import sys

sys.path.append(os.path.join(os.path.curdir, './src'))

from rotations import q2mat, eul2R
from cameras import get_intrinsics, get_extrinsics, project, unproject

class TestCameraFunctions:
    def test_get_intrinsics(self):
        f = 800
        c_x = 320
        c_y = 240
        expected_K = np.array([
            [-f,   0, c_x],
            [  0, f, c_y],
            [  0,   0,    1]
        ])
        expected_K_homogeneous = np.hstack([expected_K, np.zeros((3,1))])
        K = get_intrinsics(f, c_x, c_y)
        assert np.array_equal(K, expected_K_homogeneous), "Intrinsic matrix K is incorrect."

    def test_get_extrinsics_identity(self):
        # Quaternion representing no rotation [w, x, y, z]
        q = np.array([1, 0, 0, 0])
        p = np.array([0, 0, 0])
        # R_q should be identity
        # R_0 is a rotation of 0 around x, pi around y, -pi/2 around z
        expected_R = np.eye(3)
        expected_t = -expected_R @ p
        expected_transformation = np.vstack([
            np.hstack([expected_R, expected_t.reshape(3,1)]),
            [0, 0, 0, 1]
        ])
        transformation = get_extrinsics(q, p)
        print(transformation)
        print(expected_transformation)
        assert np.allclose(transformation, expected_transformation), "Extrinsic matrix is incorrect for identity quaternion and zero position."

    def test_get_extrinsics_rotation_translation(self):
        # 90 degree rotation around z-axis
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        p = np.array([1, 2, 3])
        R_q = q2mat(q)
        R_0 = eul2R(0, np.pi, -np.pi/2)
        R = R_q @ R_0
        t = -R @ p
        expected_transformation = np.vstack([
            np.hstack([R, t.reshape(3,1)]),
            [0, 0, 0, 1]
        ])
        transformation = get_extrinsics(q, p)
        assert np.allclose(transformation, expected_transformation), "Extrinsic matrix is incorrect for given rotation and translation."

    def test_project_simple(self):
        # Simple projection with identity intrinsic matrix
        P = np.hstack([np.eye(3), np.zeros((3,1))])
        points_3d = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        expected_2d = points_3d[:, :2] / points_3d[:, 2:3]
        projected = project(P, points_3d, z_clip=True)
        assert np.allclose(projected, expected_2d), "Projection with identity intrinsic matrix failed."

    def test_project_z_clip(self):
        P = np.hstack([np.eye(3), np.zeros((3,1))])
        points_3d = np.array([
            [1, 1, 1],
            [2, 2, -2],  # Should be clipped
            [3, 3, 3]
        ])
        expected_2d = points_3d[points_3d[:,2] > 0, :2] / points_3d[points_3d[:,2] > 0, 2:3]
        projected = project(P, points_3d, z_clip=True)
        assert np.allclose(projected, expected_2d), "Z-clipping in projection failed."

    def test_unproject_simple(self):
        # Unproject with identity intrinsic matrix
        P = np.hstack([np.eye(3), np.zeros((3,1))])
        points_2d = np.array([
            [1, 1],
            [2, 2],
            [3, 3]
        ])
        expected_3d = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])
        unprojected = unproject(P, points_2d)
        print(unprojected)
        assert np.allclose(unprojected, expected_3d), "Unprojection with identity intrinsic matrix failed."

    def test_unproject_invalid_P(self):
        # Test unproject with a singular matrix P
        P = np.zeros((3,4))
        points_2d = np.array([[1, 1]])
        with pytest.raises(np.linalg.LinAlgError):
            unproject(P, points_2d)

    def test_project_unproject_consistency(self):
        # Test that unprojecting then projecting returns the original points
        P = get_intrinsics(800, 320, 240)
        points_3d = np.random.rand(10, 3) + 1  # Ensure z > 0
        projected = project(P, points_3d)
        unprojected = unproject(P, projected)
        # Since unprojection maps to a canonical 3D space, re-projection might not return exactly original
        # Instead, we can check if the projections are consistent
        reprojected = project(P, unprojected)
        assert np.allclose(projected, reprojected, atol=1e-6), "Projection-Unprojection consistency failed."

    def test_unproject_project_consistency(self):
        # Test that projecting then unprojecting returns points on the same ray
        P = get_intrinsics(800, 320, 240)
        points_3d = np.random.rand(10, 3) + 1  # Ensure z > 0
        projected = project(P, points_3d)
        unprojected = unproject(P, projected)
        # Each unprojected point should lie on the ray defined by the original 3D point
        # Check if they are scalar multiples
        for original, recon in zip(points_3d, unprojected):
            ratio = original / recon
            # All elements should have the same ratio
            assert np.allclose(ratio, ratio[0], atol=1e-6), "Unprojected points do not lie on the original ray."

if __name__ == "__main__":
    pytest.main()