import os
from utils.Phases import Phases


def join_and_create(path, folder):
    full_path = os.path.join(path, folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    return full_path


def path_to_datasets(dataset="megadepth"):
    datasetsPath = f'../datasets/{dataset}'
    os.makedirs(datasetsPath, exist_ok=True)
    return datasetsPath


def path_to_images(conf):  # not in use
    return ""

def path_to_condition(conf):
    experiments_folder = os.path.join('..', 'results')
    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    exp_name = conf.get_string('exp_name')
    return join_and_create(experiments_folder, exp_name)


def path_to_exp(conf):
    exp_ver = conf.get_string('exp_version')
    exp_ver_path = join_and_create(path_to_condition(conf), exp_ver)

    return exp_ver_path


def path_to_phase(conf, phase):
    exp_path = path_to_exp(conf)
    return join_and_create(exp_path, phase.name)


def path_to_scan(conf, phase, scan=None):
    exp_path = path_to_phase(conf, phase)
    scan = conf.get_string("dataset.scan") if scan is None else scan
    return join_and_create(exp_path, scan)


def path_to_model(conf, phase, epoch=None, scan=None, best=False):
    if conf.get_string('pretrainedPath', default=None) is not None and phase is not Phases.FINE_TUNE and best:
        return conf.get_string('pretrainedPath')

    if phase in [Phases.TRAINING, Phases.VALIDATION, Phases.TEST]:
        parent_folder = path_to_exp(conf)
    else:
        parent_folder = path_to_scan(conf, phase, scan=scan)

    if best:
        models_path = join_and_create(parent_folder, 'models')
        if epoch is None:
            modelsList = os.listdir(models_path)
            modelsList = [int(model.split('_Ep')[1].split('.pt')[0]) for model in modelsList if model.startswith('Model')]
            epoch = sorted(modelsList)[-1]
    else:
        models_path = join_and_create(parent_folder, 'models_all')

    if epoch is None:
        model_file_name = "Final_Model.pt"
    else:
        model_file_name = "Model_Ep{}.pt".format(epoch)

    return os.path.join(models_path, model_file_name)

def path_to_model_resume_optimizing(conf, epoch=None, scan=None):

    parent_folder = path_to_exp(conf)
    print(parent_folder)
    models_path = os.path.join(parent_folder, 'OPTIMIZATION', scan, 'models_all')
    modelsList = os.listdir(models_path)
    modelsList = [int(model.split('_Ep')[1].split('.pt')[0]) for model in modelsList if model.startswith('Model')]
    epoch = sorted(modelsList)[-1]
    model_file_name = "Model_Ep{}.pt".format(epoch)

    return os.path.join(models_path, model_file_name), epoch


def path_to_model_resume_learning(conf):

    parent_folder = path_to_exp(conf)
    print(parent_folder)
    # else:
    #     parent_folder = path_to_scan(conf, phase, scan=scan)

    models_path = os.path.join(parent_folder, 'models_all')
    modelsList = os.listdir(models_path)
    modelsList = [int(model.split('_Ep')[1].split('.pt')[0]) for model in modelsList if model.startswith('Model')]
    epoch = sorted(modelsList)[-1]
    model_file_name = "Model_Ep{}.pt".format(epoch)

    return os.path.join(models_path, model_file_name), epoch

def path_to_learning_data(conf, phase):
    return join_and_create(path_to_condition(conf), phase)



def path_to_cameras(conf, phase, epoch=None, scan=None,round=""):
    scan_path = path_to_exp(conf)
    cameras_path = join_and_create(scan_path, 'forFigures')
    cameras_path = join_and_create(cameras_path, round)

    if epoch is None:
        cameras_file_name = f"{scan}_Final_Cameras"
    else:
        cameras_file_name = "{0}_Cameras_Ep{1}".format(scan, epoch)

    return os.path.join(cameras_path, cameras_file_name)

def path_to_outliers(conf, phase, epoch=None, scan=None):
    scan_path = path_to_scan(conf, phase, scan=scan)
    cameras_path = join_and_create(scan_path, 'outliers_results')

    if epoch is None:
        cameras_file_name = "Final_outliers"
    else:
        cameras_file_name = "outliers_Ep{}".format(epoch)

    return os.path.join(cameras_path, cameras_file_name)


def path_to_metrics(conf, phase, epoch=None, scan=None):
    scan_path = path_to_scan(conf, phase, scan=scan)
    excels_path = join_and_create(scan_path, 'metrics')

    if epoch is None:
        metrics_file_name = "Final_metrics"
    else:
        metrics_file_name = "metrics_Ep{}".format(epoch)

    metrics_file_name += '.xlsx'

    return os.path.join(excels_path, metrics_file_name)

def path_to_plots(conf, phase, epoch=None, scan=None, triangulation=False, filtered=False, extraText=None):
    scan_path = path_to_scan(conf, phase, scan=scan)
    plots_path = join_and_create(scan_path, 'plots')

    if epoch is None:
        plots_file_name = "Final_plots.html"
    else:
        plots_file_name = "Plot_Ep{}.html".format(epoch)

    if extraText is not None:
        plots_file_name = plots_file_name.split(".html")[0]+f"{extraText}.html"

    if triangulation:
        plots_file_name = plots_file_name.split(".html")[0]+"_triangulation.html"

    if filtered:
        plots_file_name = plots_file_name.split(".html")[0]+"_filtered4.html"

    return os.path.join(plots_path, plots_file_name)


def path_to_reconstructions(conf, phase, epoch=None, scan=None, name=None):
    scan_path = path_to_scan(conf, phase, scan=scan) + '_ba'
    reconstructions_path = join_and_create(scan_path, 'colmap_reconstructions')

    if epoch is None:
        plots_file_name = "Final_recon"
    else:
        plots_file_name = "recon_Ep{}".format(epoch)

    reconstructions_path = join_and_create(reconstructions_path, plots_file_name)
    if name is not None:
        reconstructions_path = join_and_create(reconstructions_path, name)

    return reconstructions_path


def path_to_wandb_logs(conf, phase, scan=None):
    if phase is Phases.OPTIMIZATION:
        exp_path = path_to_scan(conf, phase, scan=None)
    else:
        exp_path = path_to_exp(conf)
    logs_path = join_and_create(exp_path, "wandb")
    return logs_path

def path_to_logs(conf, phase):
    phase_path = path_to_phase(conf, phase)
    logs_path = join_and_create(phase_path, "logs")
    return logs_path


def path_to_code_logs(conf):
    exp_path = path_to_exp(conf)
    code_path = join_and_create(exp_path, "code")
    return code_path


def path_to_conf(conf_file):
    return os.path.join('confs', conf_file)

def get_lsf_file_path(conf, args):
    return os.path.join(path_to_exp(conf), f'{args.run_name}.lsf')

def get_lsf_output_path(conf):
    return os.path.join(path_to_exp(conf), f'output_%J_%I.txt'), os.path.join(path_to_exp(conf), f'err_%J_%I.txt')


def images_dir_for_scene(conf, scene_name: str) -> str:
    root = conf.get_string("dataset.images_root")
    base = scene_name.replace(".npz", "")
    cand = os.path.join(root, f"{base} images")
    if not os.path.isdir(cand):
        raise FileNotFoundError(f"Images dir not found: {cand}")
    return cand

def superpoint_desc_path(conf, scene_name: str, image_fname: str) -> str:
    """
    Where to find a precomputed dense descriptor tensor for this image.
    Default: <same folder as image>/<image_fname>.spdesc.pt
    If dataset.superpoint.desc_dir is set, use <desc_dir>/<scene>/<image_fname>.spdesc.pt
    """
    base_dir = conf.get_string("dataset.superpoint.desc_dir", "")
    if base_dir:
        # e.g., desc_dir/5016 new images/<filename>.spdesc.pt
        scene_dir = images_dir_for_scene(conf, scene_name)
        leaf = f"{os.path.basename(image_fname)}.spdesc.pt"
        return os.path.join(base_dir, os.path.basename(scene_dir), leaf)
    else:
        return os.path.join(os.path.dirname(image_fname), os.path.basename(image_fname) + ".spdesc.pt")