import torch
from utils import geo_utils, dataset_utils, sparse_utils
from datasets import  Euclidean
import os.path
from pyhocon import ConfigFactory
import numpy as np
import warnings
from utils import dataset_utils as du

class SceneData:
    def __init__(self, M, Ns, Ps_gt, scan_name, dilute_M=False, outliers=None, dict_info=None, nameslist=None, M_original=None, x_prebuilt=None):

        if M_original is None:
            M_original = M.detach().clone()

        # Dilute M
        if dilute_M:
            M = geo_utils.dilutePoint(M)

        n_images = Ps_gt.shape[0]

        # Set attribute
        self.scan_name = scan_name
        self.y = Ps_gt
        self.M = M
        self.M_original = M_original
        self.Ns = Ns
        self.outlier_indices = outliers

        # M to sparse matrix (use prebuilt if provided)
        if x_prebuilt is not None:
            self.x = x_prebuilt
        else:
            self.x = dataset_utils.M2sparse(M, normalize=True, Ns=Ns, M_original=M_original)
            # identity (x uses RAW column ids; this is kept only for compatibility)
            self.raw_cols_kept = torch.arange(M.shape[1], dtype=torch.long)

        # Get image list
        if nameslist is None:
            self.img_list = torch.arange(n_images)
        else:
            self.img_list = nameslist



        # Prepare Ns inverse transpose
        self.Ns_invT = torch.transpose(torch.inverse(Ns), 1, 2)

        # Get valid points
        self.valid_pts = dataset_utils.get_M_valid_points(M)

        # Normalize M
        self.norm_M = geo_utils.normalize_M(M, Ns, self.valid_pts).transpose(1, 2).reshape(n_images * 2, -1)


        # Stats of the scene
        self.dict_info = dict_info


    def to(self, *args, **kwargs):
        for key in self.__dict__:
            if not key.startswith('__'):
                attr = getattr(self, key)
                if isinstance(attr, sparse_utils.SparseMat) or torch.is_tensor(attr):
                    setattr(self, key, attr.to(*args, **kwargs))

        return self


def create_scene_data(conf, phase=None):
    # Init
    scan = conf.get_string('dataset.scan')
    calibrated = conf.get_bool('dataset.calibrated')
    dilute_M = conf.get_bool('dataset.diluteM', default=False)


    # Get raw data
    if calibrated:
        M, Ns, Ps_gt, outliers, dict_info, namesList, M_original = Euclidean.get_raw_data(conf, scan, phase)
    else:
        raise ValueError("The code doesn't support the uncalibrated case")

    data = SceneData(M, Ns, Ps_gt, scan, dilute_M, outliers=outliers, dict_info=dict_info, nameslist=namesList, M_original=M_original)
    du.attach_superpoint_features(data, conf, device=data.x.values.device)
    return data

def sample_data(data, num_samples, adjacent=True):
    """For a given scene, randomly sample num_samples cameras (rows), adjacent or not.
    Note: when the requested num_samples is more than available cameras, all cameras will be returned"""

    # Get indices
    indices = dataset_utils.sample_indices(len(data.y), num_samples, adjacent=adjacent)
    M_indices = np.sort(np.concatenate((2 * indices, 2 * indices + 1)))

    indices = torch.from_numpy(indices).squeeze()
    M_indices = torch.from_numpy(M_indices).squeeze()

    # slice dense tensors
    y  = data.y[indices]
    Ns = data.Ns[indices]
    M_rows = data.M[M_indices]
    outlier_indices = data.outlier_indices[indices]

    col_keep_raw = (M_rows > 0).sum(dim=0) > 2
    keep_raw = torch.nonzero(col_keep_raw, as_tuple=False).squeeze(1).long()

    outlier_indices = outlier_indices[:, col_keep_raw]
    M = M_rows[:, col_keep_raw]

    if torch.is_tensor(data.img_list):
        names_sel = data.img_list[indices]
    else:
        names_sel = [data.img_list[i] for i in indices.tolist()]

    # ================= slice EXISTING sparse x (no rebuild) =================
    x_parent = data.x
    ii_old = x_parent.indices[0]
    jj_old = x_parent.indices[1]
    feat   = x_parent.values.shape[1]

    # map old camera id -> new camera id (or -1 if not sampled)
    m_old = x_parent.shape[0]
    old_to_new_i = torch.full((m_old,), -1, dtype=torch.long, device=ii_old.device)
    for new_i, old_i in enumerate(indices.tolist()):
        old_to_new_i[old_i] = new_i

    # for every row in parent x, RAW column id is just jj_old
    raw_j_all = jj_old

    # which rows belong to sampled cameras?
    i_new_all = old_to_new_i[ii_old]
    cam_kept_for_row = i_new_all >= 0

    # which rows belong to RAW columns we keep in the sampled scene?
    col_kept_for_row = col_keep_raw.to(raw_j_all.device)[raw_j_all]

    # observation must actually exist for that sampled (cam, raw_col)
    obs_for_row = torch.zeros_like(cam_kept_for_row, dtype=torch.bool)
    sel_basic = cam_kept_for_row & col_kept_for_row
    if sel_basic.any():
        i_new_sel = i_new_all[sel_basic].cpu()
        rj_sel    = raw_j_all[sel_basic].cpu()
        u = M_rows[2 * i_new_sel + 0, rj_sel]
        v = M_rows[2 * i_new_sel + 1, rj_sel]
        obs_for_row[sel_basic] = ((u != 0) | (v != 0)).to(obs_for_row.device)

    keep_rows = sel_basic & obs_for_row

    if not keep_rows.any():
        sampled_x = sparse_utils.SparseMat(
            values=torch.zeros((0, feat), dtype=x_parent.values.dtype, device=x_parent.values.device),
            indices=torch.zeros((2, 0), dtype=torch.long, device=x_parent.indices.device),
            cam_per_pts=torch.zeros((0, 1), dtype=torch.long, device=x_parent.cam_per_pts.device),
            pts_per_cam=torch.zeros((len(indices), 1), dtype=torch.long, device=x_parent.pts_per_cam.device),
            shape=(len(indices), 0, feat),
        )
    else:
        # keep rows
        vals_keep   = x_parent.values[keep_rows]
        raw_j_keep  = raw_j_all[keep_rows]

        # remap cameras to [0..m_sample-1]
        ii_new = i_new_all[keep_rows]

        # remap RAW columns to compact [0..n_sample-1] in the SAME RAW order as M
        raw_to_newpos = -torch.ones((data.M.shape[1],), dtype=torch.long, device=raw_j_all.device)
        raw_to_newpos[keep_raw.to(raw_j_all.device)] = torch.arange(
            keep_raw.numel(), dtype=torch.long, device=raw_j_all.device
        )
        jj_new = raw_to_newpos[raw_j_keep]

        # build counts
        m_new = len(indices)
        n_new = keep_raw.numel()
        cam_per_pts = torch.bincount(jj_new, minlength=n_new).view(-1, 1)
        pts_per_cam = torch.bincount(ii_new, minlength=m_new).view(-1, 1)

        new_indices = torch.stack([ii_new, jj_new], dim=0)

        sampled_x = sparse_utils.SparseMat(
            values=vals_keep,
            indices=new_indices,
            cam_per_pts=cam_per_pts.to(x_parent.cam_per_pts.device),
            pts_per_cam=pts_per_cam.to(x_parent.pts_per_cam.device),
            shape=(m_new, n_new, feat),
        )
    # ======================================================================

    sampled_data = SceneData(
        M, Ns, y, data.scan_name,
        outliers=outlier_indices,
        nameslist=names_sel,
        M_original=data.M_original,
        x_prebuilt=sampled_x,           # << use prebuilt sparse
    )

    if (sampled_data.x.pts_per_cam == 0).any():
        warnings.warn('Cameras with no points for dataset '+ data.scan_name)

    return sampled_data


def create_scene_data_from_list(scan_names_list, conf):
    data_list = []
    for scan_name in scan_names_list:
        conf["dataset"]["scan"] = scan_name
        data = create_scene_data(conf)
        data_list.append(data)

    return data_list


def test_data(data, conf):
    import loss_functions

    # Test Losses of GT and random on data
    repLoss = loss_functions.ESFMLoss(conf)
    cams_gt = prepare_cameras_for_loss_func(data.y, data)
    cams_rand = prepare_cameras_for_loss_func(torch.rand(data.y.shape), data)

    print("Loss for GT: Reprojection = {}".format(repLoss(cams_gt, data)))
    print("Loss for rand: Reprojection = {}".format(repLoss(cams_rand, data)))


def prepare_cameras_for_loss_func(Ps, data):
    Vs_invT = Ps[:, 0:3, 0:3]
    Vs = torch.inverse(Vs_invT).transpose(1, 2)
    ts = torch.bmm(-Vs.transpose(1, 2), Ps[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()
    pts_3D = torch.from_numpy(geo_utils.n_view_triangulation(Ps.numpy(), data.M.numpy(), data.Ns.numpy())).float()
    return {"Ps": torch.bmm(data.Ns, Ps), "pts3D": pts_3D}


def get_subset(data, subset_size):
    # Get subset indices
    valid_pts = dataset_utils.get_M_valid_points(data.M)
    n_cams = valid_pts.shape[0]

    first_idx = valid_pts.sum(dim=1).argmax().item()
    curr_pts = valid_pts[first_idx].clone()
    valid_pts[first_idx] = False
    indices = [first_idx]

    for i in range(subset_size - 1):
        shared_pts = curr_pts.expand(n_cams, -1) & valid_pts
        next_idx = shared_pts.sum(dim=1).argmax().item()
        curr_pts = curr_pts | valid_pts[next_idx]
        valid_pts[next_idx] = False
        indices.append(next_idx)

    print("Cameras are:")
    print(indices)

    indices = torch.sort(torch.tensor(indices))[0]
    M_indices = torch.sort(torch.cat((2 * indices, 2 * indices + 1)))[0]
    y, Ns = data.y[indices], data.Ns[indices]
    M = data.M[M_indices]
    M = M[:, (M > 0).sum(dim=0) > 2]
    return SceneData(M, Ns, y, data.scan_name + "_{}".format(subset_size), outliers=data.outlier_indices[indices], dict_info=data.dict_info, nameslist=data.img_list[indices])

if __name__ == "__main__":
    test_dataset()

