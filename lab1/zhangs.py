import os
import numpy as np
import cv2

from scipy.optimize import least_squares
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def find_homography(world_pts, image_pts):
    """
    Direct Linear Transform (DLT) for homography estimation.
    Solves H such that image_pts ≈ H * world_pts
    """
    world_pts = np.asarray(world_pts, dtype=np.float64)
    image_pts = np.asarray(image_pts, dtype=np.float64)
    n = world_pts.shape[0]

    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        X, Y = world_pts[i]
        u, v = image_pts[i]
        A[2 * i] = [-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, v * X, v * Y, v]

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def get_homography_constraints(H):
    h0 = H[:, 0]
    h1 = H[:, 1]

    v00 = np.array([
        h0[0] * h0[0],
        h0[0] * h0[1] + h0[1] * h0[0],
        h0[1] * h0[1],
        h0[2] * h0[0] + h0[0] * h0[2],
        h0[2] * h0[1] + h0[1] * h0[2],
        h0[2] * h0[2]
    ])

    v01 = np.array([
        h0[0] * h1[0],
        h0[0] * h1[1] + h0[1] * h1[0],
        h0[1] * h1[1],
        h0[2] * h1[0] + h0[0] * h1[2],
        h0[2] * h1[1] + h0[1] * h1[2],
        h0[2] * h1[2]
    ])

    v11 = np.array([
        h1[0] * h1[0],
        h1[0] * h1[1] + h1[1] * h1[0],
        h1[1] * h1[1],
        h1[2] * h1[0] + h1[0] * h1[2],
        h1[2] * h1[1] + h1[1] * h1[2],
        h1[2] * h1[2]
    ])

    return v01, v00 - v11


def solve_for_intrinsics(V):
    """
    Solve for the camera intrinsics from the constraint matrix V.

    V * b = 0, where b = [B11, B12, B22, B13, B23, B33]^T

    Returns the B matrix and the intrinsic matrix K.
    """
    print(f"Solving for intrinsics from V matrix of shape {V.shape}")

    U, s, Vt = np.linalg.svd(V)
    print(f"SVD singular values: {s[:3]}... (showing first 3)")

    b = Vt[-1, :]
    print(f"Raw b vector: {b}")

    # Normalize so that B33 = 1 (b[5] corresponds to B33)
    if abs(b[5]) > 1e-12:
        b = b / b[5]
    print(f"Normalized b vector (B33=1): {b}")

    if b[0] < 0:
        b = -b
        print(f"Flipped sign for positive B11: {b}")

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ], dtype=np.float64)

    print(f"B matrix:\n{B}")
    print(f"B eigenvalues: {np.linalg.eigvals(B)}")

    # Check if B is positive definite
    eigenvals = np.linalg.eigvals(B)
    min_eigenval = np.min(eigenvals)
    print(f"Minimum eigenvalue of B: {min_eigenval}")

    if min_eigenval < 0:
        print("Warning: B matrix is not positive definite, adding regularization")

        regularization = abs(min_eigenval) + 1e-6
        B += regularization * np.eye(3)
        print(f"Regularized B matrix:\n{B}")
        print(f"Regularized B eigenvalues: {np.linalg.eigvals(B)}")

    # B = K^{-T} * K^{-1}, so K^{-1} = chol(B)
    try:
        # Cholesky decomposition of B gives L such that B = L * L^T
        # Since B = (K^{-1})^T * (K^{-1}), we have K^{-1} = L^T
        L = np.linalg.cholesky(B, upper=False)
        K_inv = L.T

        K = np.linalg.inv(K_inv)
        K = K / K[2, 2]

        print(f"Computed K matrix:\n{K}")
        return K

    except np.linalg.LinAlgError as e:
        print(f"Cholesky failed: {e}, trying eigenvalue decomposition")

        eigenvals, eigenvecs = np.linalg.eigh(B)
        if np.all(eigenvals > 0):
            # B = Q * Λ * Q^T, so sqrt(B) = Q * sqrt(Λ) * Q^T
            sqrt_eigenvals = np.sqrt(eigenvals)
            K_inv = eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
            K = np.linalg.inv(K_inv)
            K = K / K[2, 2]
            print(f"Eigenvalue decomposition result:\n{K}")
            return K
        else:
            raise RuntimeError(f"Both Cholesky and eigenvalue methods failed. B matrix eigenvalues: {eigenvals}")


def project_points(K, R, t, pts_3d):
    """Projects Nx3 points to image plane using K, R, t. Returns Nx2."""
    pts_3d = np.asarray(pts_3d, dtype=np.float64)

    pts_cam = (R @ pts_3d.T).T + t.reshape(1, 3)
    pts_proj_h = (K @ pts_cam.T).T
    pts_proj = pts_proj_h[:, :2] / pts_proj_h[:, 2:3]
    return pts_proj


def compute_extrinsics_from_H(H, K):
    """From a homography and K, compute R (3x3) and t (3,) for that view."""
    Kinv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    # scale factor
    lam = 1.0 / np.linalg.norm(Kinv @ h1)
    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    t = lam * (Kinv @ h3)
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    return R, t


def zhang_linear_calibration(image_points_list, world_points_2d, image_size=None):
    if len(image_points_list) < 3:
        raise ValueError("Need at least 3 images for calibration")

    print(f"Processing {len(image_points_list)} images with {len(world_points_2d)} points each")
    print(
        f"World coordinate range: X=[{world_points_2d[:, 0].min():.1f}, {world_points_2d[:, 0].max():.1f}], Y=[{world_points_2d[:, 1].min():.1f}, {world_points_2d[:, 1].max():.1f}]")

    homographies = []
    V_constraints = []

    for i, image_points in enumerate(image_points_list):
        print(f"Computing homography for image {i + 1}")

        H = find_homography(world_points_2d, image_points)
        H /= H[2, 2]  # Normalize so H[2,2] = 1

        homographies.append(H)

        v01, v_diff = get_homography_constraints(H)
        V_constraints.extend([v01, v_diff])

    V = np.array(V_constraints, dtype=np.float64)
    print(f"V matrix shape: {V.shape} (should be 2*{len(image_points_list)} x 6)")

    cond_num = np.linalg.cond(V)
    print(f"{cond_num=:.2e}")

    K = solve_for_intrinsics(V)

    if image_size is not None:
        height, width = image_size
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        print("\nValidation:")
        print(f"Focal lengths: fx={fx:.1f}, fy={fy:.1f} pixels")
        print(f"Principal point: cx={cx:.1f}, cy={cy:.1f} pixels")
        print(f"Expected principal point range: cx ~ {width // 2}, cy ~ {height // 2}")

        reasonable_focal = 0.1 * max(width, height) < min(fx, fy) < 5.0 * max(width, height)
        reasonable_center = 0.2 * width < cx < 0.8 * width and 0.2 * height < cy < 0.8 * height

        if reasonable_focal and reasonable_center:
            print("SUCCESS: Calibration results appear reasonable")
        else:
            print("WARNING: Calibration results may be unreliable")
            if not reasonable_focal:
                print("  - Focal lengths seem unrealistic")
            if not reasonable_center:
                print("  - Principal point is outside expected range")

    return K, homographies


def calibrate_from_chessboards(images,
                               pattern_size=(10, 6),
                               refine_with_lm=True):
    pattern_w, pattern_h = pattern_size

    world_points_2d = []
    for y in range(pattern_h):
        for x in range(pattern_w):
            world_points_2d.append([x, y])
    world_points_2d = np.array(world_points_2d, dtype=np.float64)
    world_pts_3d = np.hstack([world_points_2d, np.zeros((len(world_points_2d), 1))])

    image_pts_list = []
    good_img_indices = []
    for idx, img in enumerate(images):
        ret, corners = cv2.findChessboardCorners(img, (pattern_w, pattern_h),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret or corners is None:
            print(f"[Warning] Chessboard not found in image {idx}, skipping.")
            continue
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        corners_refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), term)
        ip = corners_refined.reshape(-1, 2)
        image_pts_list.append(ip)
        good_img_indices.append(idx)

    if len(image_pts_list) < 3:
        raise RuntimeError(f"Need at least 3 good views. Only found {len(image_pts_list)} valid images.")

    print(f"Successfully processed {len(image_pts_list)} out of {len(images)} images for calibration.")

    print("\n=== Zhang's Linear Calibration ===")
    K_zhang, homographies_zhang = zhang_linear_calibration(
        image_pts_list, world_points_2d,
        image_size=images[0].shape[:2] if images else None
    )
    print(f"Zhang's linear method completed.")

    if refine_with_lm:
        print("\n=== Nonlinear Refinement (Levenberg-Marquardt) ===")
        print("Refining intrinsics using LM optimization...")

        fx0 = K_zhang[0, 0]
        fy0 = K_zhang[1, 1]
        cx0 = K_zhang[0, 2]
        cy0 = K_zhang[1, 2]

        extrinsics_lm = []
        for H in homographies_zhang:
            try:
                R, t = compute_extrinsics_from_H(H, K_zhang)  # Use linear K as initial guess
                extrinsics_lm.append((R, t))
            except:
                extrinsics_lm.append((np.eye(3), np.zeros(3)))

        params = np.array([fx0, fy0, cx0, cy0], dtype=np.float64)
        for R, t in extrinsics_lm:
            rvec, _ = cv2.Rodrigues(R)
            params = np.hstack([params, rvec.ravel(), t.ravel()])

        def reprojection_residuals(p):
            fx, fy, cx, cy = p[:4]
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)

            residuals = []
            offset = 4
            for i in range(len(image_pts_list)):
                rvec = p[offset:offset + 3]
                tvec = p[offset + 3:offset + 6]
                offset += 6
                R_i, _ = cv2.Rodrigues(rvec)
                proj = project_points(K, R_i, tvec, world_pts_3d)
                residuals.append((proj - image_pts_list[i]).ravel())
            return np.hstack(residuals)

        res = least_squares(reprojection_residuals, params, method='lm', verbose=1, max_nfev=500)
        p_opt = res.x

        fx_opt, fy_opt, cx_opt, cy_opt = p_opt[:4]
        K_refined = np.array([
            [fx_opt, 0, cx_opt],
            [0, fy_opt, cy_opt],
            [0, 0, 1]
        ], dtype=np.float64)

        residuals = reprojection_residuals(p_opt)
        rms_refined = np.sqrt(np.mean(residuals ** 2))

        print(f"Refined K matrix:\n{K_refined}")
        print(f"RMS reprojection error: {rms_refined:.3f} pixels")

        rms_initial = np.sqrt(np.mean(reprojection_residuals(params) ** 2))
        improvement = rms_initial - rms_refined
        if improvement > 0.01:
            print(f"Refinement improved RMS by {improvement:.3f} pixels")
            K_final = K_refined
            method_final = "Zhang LM refined"
        else:
            print("Refinement did not significantly improve results")
            K_final = K_zhang
            method_final = "Zhang Linear"
            rms_refined = rms_initial
    else:
        K_refined = None
        K_final = K_zhang
        method_final = "Zhang Linear"
        rms_refined = None

    print("\n=== CALIBRATION SUMMARY ===")
    print(f"Method used: {method_final}")
    if rms_refined is not None:
        print(f"RMS reprojection error: {rms_refined:.3f} pixels")
    print("Final K matrix:")
    print(K_final)

    return {
        'K_linear': K_zhang,
        'K_refined': K_refined,
        'K_final': K_final,
        'method_final': method_final,
        'rms_refined': rms_refined,
        'homographies': homographies_zhang,
        'image_points': image_pts_list,
        'world_points_2d': world_points_2d,
        'world_points_3d': world_pts_3d,
        'pattern_size': pattern_size,
    }


if __name__ == "__main__":
    images_dir = './new_images'
    images = []
    filenames = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for f in filenames:
        img = cv2.imread(os.path.join(images_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Failed to read", f)
            continue
        images.append(img)

    result = calibrate_from_chessboards(images, pattern_size=(10, 6), refine_with_lm=True)

    if len(result['image_points']) > 0:
        K = result['K_final']
        method = result['method_final']
        rms_final = result['rms_refined']

        H0 = result['homographies'][0]
        R0, t0 = compute_extrinsics_from_H(H0, K)

        reproj0 = project_points(K, R0, t0, result['world_points_3d'])
        img0 = images[0].copy()
        plt.figure(figsize=(8, 6))
        plt.imshow(img0, cmap='gray')

        det = result['image_points'][0]
        plt.scatter(det[:, 0], det[:, 1], s=20, label='detected', marker='o')
        plt.scatter(reproj0[:, 0], reproj0[:, 1], s=10, label='reprojected', marker='x')
        plt.legend()
        plt.title(f"First image: detected vs reprojection ({method})")
        if rms_final is not None:
            plt.title(f"First image: detected vs reprojection ({method}, RMS: {rms_final:.3f}px)")
        plt.show()

        print(f"\nCalibration method: {method}")
        print("Final K matrix:")
        print(result['K_final'])
