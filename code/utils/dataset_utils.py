import torch
from utils import geo_utils, general_utils, sparse_utils, plot_utils
from utils.Phases import Phases
import numpy as np
import networkx as nx
import os
import torch.nn.functional as F
from utils import path_utils
from utils.sparse_utils import SparseMat

def is_valid_sample(data, min_pts_per_cam=10, phase=Phases.TRAINING):
    if phase is Phases.TRAINING:
        return data.x.pts_per_cam.min().item() >= min_pts_per_cam
    else:
        return True


def divide_indices_to_train_test(N, n_val, n_test=0):
    perm = np.random.permutation(N)
    test_indices = perm[:n_test] if n_test>0 else []
    val_indices = perm[n_test:n_test+n_val]
    train_indices = perm[n_test+n_val:]
    return train_indices, val_indices, test_indices


def sample_indices(N, num_samples, adjacent):
    if num_samples == 1:  # Return all the data
        indices = np.arange(N)
    else:
        if num_samples < 1:  # fraction
            num_samples = int(np.ceil(num_samples * N))
        num_samples = max(2, num_samples)
        if num_samples >= N:
            return np.arange(N)
        if adjacent:
            start_ind = np.random.randint(0,N-num_samples+1)
            end_ind = start_ind+num_samples
            indices = np.arange(start_ind, end_ind)
        else:
            indices = np.random.choice(N,num_samples,replace=False)
    return indices


def save_cameras(outputs, conf, curr_epoch, phase):
    xs = outputs['xs']
    M = geo_utils.xs_to_M(xs)
    general_utils.save_camera_mat(conf, outputs, outputs['scan_name'], phase, curr_epoch)

def save_outliers(outputs, conf, curr_epoch, phase):
    if curr_epoch is None:
        general_utils.save_outliers_mat(conf, outputs, outputs['scan_name'], phase, curr_epoch)

# def save_metrics(outputs, conf, curr_epoch, phase):
#     general_utils.save_outliers_mat(conf, outputs, outputs['scan_name'], phase, curr_epoch)


def get_data_statistics(all_data, outputs=None):
    valid_pts = all_data.valid_pts
    valid_pts_stat = valid_pts.sum(dim=0).float()
    stats = {"Max_2d_pt": all_data.M.max().item(), "Num_2d_pts": valid_pts.sum().item(), "n_pts": all_data.M.shape[-1],
             "pts_per_cam_mean":  valid_pts.sum(dim=1).float().mean().item(), "Cameras_per_pts_mean": valid_pts_stat.mean().item(), "Cameras_per_pts_std": valid_pts_stat.std().item(),
             "Num of cameras": all_data.y.shape[0]}

    if outputs is not None:
        dict_info = all_data.dict_info.copy()
        dict_info.pop("outliers_pred", None)  # safely remove if exists
        stats.update(dict_info)

    return stats


def correct_matches_global(M, Ps, Ns):
    """
    This function corrects the matches using global triangulation.
    Args:
        M (torch.Tensor): A tensor of shape (N, 2), where N is the number of matches. ## correct this line
        Ps (torch.Tensor): A tensor of shape (N, 3, 4), where N is the number of cameras.
        Ns (torch.Tensor): A tensor of shape (N, 3, 3), where N is the number of cameras.
    Returns:
        xs (torch.Tensor): A tensor of shape M
    """

    # First, we get the invalid points in M.

    M_invalid_pts = np.logical_not(get_M_valid_points(M))

    # Next, we perform global triangulation to get the corrected matches.

    Xs = geo_utils.n_view_triangulation(Ps, M, Ns)
    xs = geo_utils.batch_pflat((Ps @ Xs))[:, 0:2, :]

    # Finally, we remove the invalid points from xs.

    xs[np.isnan(xs)] = 0
    # xs[np.stack((M_invalid_pts, M_invalid_pts), axis=1)] = 0
    xs = xs.reshape(M.shape)
    xs = torch.tensor(xs)
    xs = xs.transpose(0, 1).reshape(-1, xs.shape[0] // 2, 2).transpose(0, 1)
    xs[M_invalid_pts] = 0
    xs = xs.transpose(0, 1).reshape(-1, xs.shape[0] * 2).transpose(0, 1)

    return xs.numpy()



def get_M_valid_points(M):
    n_pts = M.shape[-1]

    if type(M) is torch.Tensor:
        M_valid_pts = torch.abs(M.reshape(-1, 2, n_pts)).sum(dim=1) != 0 # zero point
        M_valid_pts[:, M_valid_pts.sum(dim=0) < 2] = False  # mask out point tracks that contain only 1 point (viewed only from one camera-f)
    else:
        M_valid_pts = np.abs(M.reshape(-1, 2, n_pts)).sum(axis=1) != 0
        M_valid_pts[:, M_valid_pts.sum(axis=0) < 2] = False

    return M_valid_pts





def M2sparse(M, normalize=False, Ns=None, M_original=None, features=None):
    n_pts = M.shape[1]
    n_cams = int(M.shape[0] / 2)

    # Get indices
    valid_pts = get_M_valid_points(M)
    cam_per_pts = valid_pts.sum(dim=0).unsqueeze(1)  # [n_pts, 1]
    pts_per_cam = valid_pts.sum(dim=1).unsqueeze(1)  # [n_cams, 1]
    mat_indices = torch.nonzero(valid_pts).T  # [2, the number of points in the scene]
    # Get Values
    # reshaped_M = M.reshape(n_cams, 2, n_pts).transpose(1, 2)  # [2m, n] -> [m, 2, n] -> [m, n, 2]
    if normalize:
        norm_M = geo_utils.normalize_M(M, Ns)
        mat_vals = norm_M[mat_indices[0], mat_indices[1], :]
    else:
        mat_vals = M.reshape(n_cams, 2, n_pts).transpose(1, 2)[mat_indices[0], mat_indices[1], :]

    mat_shape = (n_cams, n_pts, 2)
    
    return sparse_utils.SparseMat(mat_vals, mat_indices, cam_per_pts, pts_per_cam, mat_shape)


def get_M_view_adjacency(M):
    """
    Calculates the view adjacency matrix from  M.

    Args:
        M: A tracks tensor (2 * num_views, num_points).

    Returns:
        view_graph_adj: A torch.Tensor of shape (num_views, num_views) representing the view adjacency matrix,
                       where view_graph_adj[i, j] is the number of shared visible points between views i and j.
    """
    view_graph_adj = torch.zeros([M.shape[0] // 2, M.shape[0] // 2], dtype=torch.int32, device=M.device)
    M_valid_pts = get_M_valid_points(M)
    for i in range(M_valid_pts.shape[0]):
        for j in range(i + 1, M_valid_pts.shape[0]):
            num_shared_points = torch.logical_and(M_valid_pts[i], M_valid_pts[j]).sum()
            view_graph_adj[i, j] = num_shared_points
            view_graph_adj[j, i] = num_shared_points

    return view_graph_adj


def check_if_M_connected(M, thr=1, return_largest_component=False, returnAll=False):
    """
    Check connectivity of the camera-point view graph derived from the visibility matrix M.

    Args:
        M (Tensor): [2m, n] binary visibility matrix.
        thr (int): Minimum number of shared points to consider a connection between views.
        return_largest_component (bool): If True, return the largest connected component.
        returnAll (bool): If True, return all connected components.

    Returns:
        bool or (bool, List[int]) or List[Set[int]] depending on flags:
            - If no flags: returns is_connected (bool)
            - If return_largest_component: returns (is_connected, largest_component)
            - If returnAll: returns list of all components
    """
    import networkx as nx
    import numpy as np

    # Get adjacency matrix of M
    view_graph_adj = get_M_view_adjacency(M)
    view_graph_adj = view_graph_adj.detach().cpu().numpy()

    # Create binary adjacency graph based on threshold
    view_graph = nx.from_numpy_array((view_graph_adj >= thr).astype(int))

    # Check overall connectivity
    connected = nx.is_connected(view_graph)

    # Extract connected and biconnected components
    components = sorted(nx.connected_components(view_graph), key=len, reverse=True)
    #biconnected_components = sorted(nx.biconnected_components(view_graph), key=len, reverse=True)

    # print(f"Component sizes (sorted): {[len(comp) for comp in components]}")
    # print(f"The graph has {len(components)} components")


    if returnAll:
        return components
    if return_largest_component:
        largest_cc = components[0] if components else []
        return connected, list(largest_cc)

    return connected

def _assign_sparse_desc_to_uv(kpts_xy: torch.Tensor,
                              desc_DN: torch.Tensor,
                              uv_pix: torch.Tensor,
                              max_r: float = 6.0,
                              k: int = 3,
                              sigma: float = 2.0) -> torch.Tensor:
    K = uv_pix.shape[0]
    D = int(desc_DN.shape[0])
    if kpts_xy.numel() == 0 or K == 0:
        return torch.zeros((K, D), device=uv_pix.device, dtype=desc_DN.dtype)

    uv = uv_pix.to(kpts_xy.device).unsqueeze(1)       # [K,1,2]
    kp = kpts_xy.unsqueeze(0)                         # [1,N,2]
    d2 = ((uv - kp) ** 2).sum(dim=-1)                 # [K,N]

    if k <= 1:
        idx = torch.argmin(d2, dim=1)
        min_d2 = torch.gather(d2, 1, idx[:, None])[:, 0]
        out = torch.zeros((K, D), device=uv.device, dtype=desc_DN.dtype)
        ok = min_d2 <= (max_r * max_r)
        if ok.any():
            out[ok] = desc_DN[:, idx[ok]].T
        return F.normalize(out, dim=-1)

    k_eff = min(k, d2.shape[1])
    d2_sorted, idx_sorted = torch.topk(d2, k_eff, dim=1, largest=False)   # [K,k]
    ok_any = d2_sorted[:, 0] <= (max_r * max_r)

    out = torch.zeros((K, D), device=uv.device, dtype=desc_DN.dtype)
    if ok_any.any():
        w = torch.exp(-d2_sorted[ok_any] / (2 * (sigma ** 2)))             # [Kok,k]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        desc_sel = desc_DN.T[idx_sorted[ok_any]]                           # [Kok,k,D]
        out[ok_any] = (w.unsqueeze(-1) * desc_sel).sum(dim=1)
    return F.normalize(out, dim=-1)


def attach_superpoint_features(scene, conf, device=None):
    """
    Attach SuperPoint descriptors by doing KNN assignment on-the-fly.
      - For each camera i, take the observations' (u,v) from the ORIGINAL M (pixel coords),
      - Load that camera's SP payload {kpts [N,2], desc [D,N], H, W},
      - Assign KNN with gaussian weights to get one descriptor per observation,
      - Concatenate to scene.x.values so feature dim becomes 2 + sp_dim.

    Notes
    -----
    * scene.img_list[i] should be the image name/path for the i-th camera IN THIS scene (full or sampled).
      We derive the SP payload path from it using path_utils.superpoint_desc_path(...).
    """
    # If _assign_sparse_desc_to_uv is defined in this module, we can call it directly.
    # Otherwise, import it explicitly:
    try:
        _assign_fn = _assign_sparse_desc_to_uv  # noqa: F821 (provided in this module)
    except NameError:
        from utils.dataset_utils import _assign_sparse_desc_to_uv as _assign_fn

    if not conf.get_bool("dataset.superpoint.enable", False):
        return

    x = scene.x
    m = x.shape[0]
    nnz = x.values.shape[0]
    if nnz == 0 or m == 0:
        return

    device = device or x.values.device
    sp_dim = conf.get_int("dataset.superpoint.dim", 256)

    # If already attached for this x (2 reproj-features + SP dim), skip.
    if x.values.shape[1] == 2 + sp_dim:
        return
    if x.values.shape[1] > 2 + sp_dim:
        raise RuntimeError(f"{scene.scan_name}: unexpected feature width {x.values.shape[1]} (already augmented?)")

    # Figure out how to map (i_samp, j_samp) -> (i_full, j_raw)
    i_samp = x.indices[0].long()
    j_samp = x.indices[1].long()

    # Camera map
    if hasattr(scene, "orig_cam_ids") and scene.orig_cam_ids is not None:
        if scene.orig_cam_ids.numel() != m:
            raise RuntimeError(f"{scene.scan_name}: orig_cam_ids length {scene.orig_cam_ids.numel()} != m {m}")
        i_full_for_cam = scene.orig_cam_ids.long().to(device if scene.orig_cam_ids.is_cuda else "cpu")
    else:
        # Full scene fallback (identity)
        i_full_for_cam = torch.arange(m, dtype=torch.long)

    # Column (track) map
    if hasattr(scene, "sampled_j_to_raw") and scene.sampled_j_to_raw is not None:
        sampled_j_to_raw = scene.sampled_j_to_raw.long()
        max_j = int(j_samp.max().item()) if j_samp.numel() > 0 else -1
        if sampled_j_to_raw.numel() < (max_j + 1):  # strict bound
            raise RuntimeError(
                f"{scene.scan_name}: sampled_j_to_raw size {sampled_j_to_raw.numel()} <= max j_samp {max_j}"
            )
        # Compressed j (in this scene) -> RAW column id
        j_raw_all = sampled_j_to_raw[j_samp]
    else:
        # Full scene fallback (identity: indices[1] are already RAW column ids)
        j_raw_all = j_samp

    # Prepare output buffer
    desc_out = torch.zeros((nnz, sp_dim), dtype=torch.float32, device=device)

    # Convenience for pixel coords
    M_orig = scene.M_original
    if isinstance(M_orig, torch.Tensor):
        M_np = M_orig.cpu().numpy()
    else:
        M_np = M_orig  # already numpy

    # Build per-camera image path list aligned to *this* scene's cameras (0..m-1)
    # scene.img_list can be a tensor or list of names; we always use the per-scene index i_samp.
    img_dir = path_utils.images_dir_for_scene(conf, scene.scan_name)
    img_paths = []
    if torch.is_tensor(scene.img_list):
        names = scene.img_list.cpu().tolist()
    else:
        names = scene.img_list
    for nm in names:
        s = str(nm).strip()
        img_paths.append(s if os.path.isabs(s) else os.path.join(img_dir, os.path.basename(s)))

    # KNN params (match your precompute defaults)
    k =  3
    sigma = 2.0
    min_r_px =  4.0
    scale_r = 0.006
    
    print("attaching superpoint features, may take a while... ", end="", flush=True)
    # Process camera by camera to avoid huge gathers
    for i_cam in range(m):
        cam_mask = (i_samp == i_cam)
        if not torch.any(cam_mask):
            continue

        row_ids = torch.nonzero(cam_mask, as_tuple=False).squeeze(1)  # indices into x rows for this cam
        # RAW track ids for these rows
        j_raw = j_raw_all[row_ids].detach().cpu().numpy().astype(np.int64)

        # Full-scene camera id (to read from M_original)
        i_full = int(i_full_for_cam[i_cam].item())

        # Pixel coords from ORIGINAL M (RAW columns)
        u = M_np[2 * i_full + 0, j_raw]
        v = M_np[2 * i_full + 1, j_raw]
        uv_pix = torch.from_numpy(np.stack([u, v], axis=1)).float().to(device)

        # Load SP payload for THIS scene's camera i_cam (path derived from scene.img_list[i_cam])
        dpath = path_utils.superpoint_desc_path(conf, scene.scan_name, img_paths[i_cam])
        if not os.path.isfile(dpath):
            raise FileNotFoundError(f"{scene.scan_name}: missing SuperPoint payload for cam {i_cam}: {dpath}")
        payload = torch.load(dpath, map_location="cpu")

        kpts = payload["kpts"].float().to(device)  # [N,2]
        desc = payload["desc"].float().to(device)  # [D,N]
        desc = torch.nn.functional.normalize(desc, dim=0)

        # Adaptive radius by long side, with a minimum
        H = int(payload.get("H", 0)); W = int(payload.get("W", 0))
        long_side = max(H, W) or 1600
        adapt_r = max(min_r_px, scale_r * long_side)

        # Assign descriptors to the observation pixels
        D_ij = _assign_fn(kpts, desc, uv_pix, max_r=adapt_r, k=k, sigma=sigma)  # [K,D]
        if D_ij.shape[1] != sp_dim:
            raise RuntimeError(f"{scene.scan_name}: SP dim {D_ij.shape[1]} != expected {sp_dim}")

        desc_out[row_ids] = D_ij.to(device=device, dtype=torch.float32)

        # if ((i_cam + 1) % 10 == 0) or (i_cam + 1 == m):
        #     print(f"[SP attach] {scene.scan_name}: cam {i_cam+1}/{m}  nnz_cam={row_ids.numel()}")

    # Concatenate with existing x.values
    new_vals = torch.cat([x.values.to(device), desc_out.to(x.values.dtype)], dim=-1)  # [nnz, 2 + sp_dim]
    new_shape = (x.shape[0], x.shape[1], new_vals.shape[1])
    scene.x = SparseMat(new_vals, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)