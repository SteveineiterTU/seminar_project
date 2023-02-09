import numpy as np
import torch
import os
import pickle
import argparse
from easydict import EasyDict
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

import unets
from diffusion import GuassianDiffusion
from helpers import fix_legacy_dict, check_folder, visualize_gt, visualize_progress, save_files, \
    get_filepaths_from_dir, read_image

from scipy.optimize import minimize

# Hyperparameters
LAMBDA2 = 0.2
LAMBDA3 = 0.001
RANDOM_SEED = 1000


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        "-name",
        type=str,
        required=True,
        help="the name of the current experiment (used to set up the save_dir)",
    )
    parser.add_argument(
        "--gan_model_dir",
        "-gdir",
        type=str,
        required=True,
        help="directory for the Victim GAN model",
    )
    parser.add_argument(
        "--pos_data_dir",
        "-posdir",
        type=str,
        help="the directory for the positive (training) query images set",
    )
    parser.add_argument(
        "--neg_data_dir",
        "-negdir",
        type=str,
        help="the directory for the negative (testing) query images set",
    )
    parser.add_argument(
        "--data_num",
        "-dnum",
        type=int,
        default=5,
        help="the number of query images to be considered",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=1,
        help="batch size (should not be too large for better optimization performance)",
    )
    parser.add_argument(
        "--resolution",
        "-resolution",
        type=int,
        default=64,
        help="generated image resolution",
    )
    parser.add_argument(
        "--initialize_type",
        "-init",
        type=str,
        default="random",
        choices=[
            "zero",  # 'zero': initialize the z to be zeros
            "random",  # 'random': use normal distributed initialization
            "nn",  # 'nn': use the result of the knn as the initialization
        ],
        help="the initialization techniques",
    )
    parser.add_argument(
        "--nn_dir",
        "-ndir",
        type=str,
        help="the directory for storing the fbb(KNN) results",
    )
    parser.add_argument(
        "--distance",
        "-dist",
        type=str,
        default="l2-lpips",
        choices=["l2", "l2-lpips"],
        help="the objective function type",
    )
    parser.add_argument(
        "--if_norm_reg",
        "-reg",
        action="store_true",
        default=True,
        help="enable the norm regularizer",
    )
    parser.add_argument(
        "--maxfunc",
        "-mf",
        type=int,
        default=10,
        help="the maximum number of function calls (for scipy optimizer)",
    )

    # ================================= Arguments for diffusion =================================
    parser.add_argument("--device", "-device", type=str, default="cuda:0")
    parser.add_argument(
        "--sampling_steps",
        "-sampling_steps",
        type=int,
        default=10,
        help="The sampling steps used in the reverse process. "
             "Since we are using DDIM we can use less steps",
    )
    parser.add_argument(
        "--diffusion_steps",
        "-diffusion_steps",
        type=int,
        default=1000,
        help="The diffusion steps used in the training of the diffusion model.",
    )

    return parser.parse_args()


def check_args(args):
    """
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    """
    ## load dir
    assert os.path.exists(args.gan_model_dir)

    ## set up save_dir
    save_dir = os.path.join(os.path.dirname(__file__), "results/pbb", args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, "params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(
        vars(args), open(os.path.join(save_dir, "params.pkl"), "wb"), protocol=2
    )

    return args, save_dir, args.gan_model_dir


#############################################################################################################
# main optimization function
#############################################################################################################
class LatentZ(torch.nn.Module):
    def __init__(self, init_val):
        super(LatentZ, self).__init__()
        self.z = torch.nn.Parameter(init_val.data)

    def forward(self):
        return self.z

    def reinit(self, init_val):
        self.z = torch.nn.Parameter(init_val.data)


class Loss(torch.nn.Module):
    def __init__(self, model, distance, if_norm_reg=False, z_dim=100):
        super(Loss, self).__init__()
        self.distance = distance
        self.lpips_model = None  # Need to create a model if we want to use it.
        self.model = model
        self.if_norm_reg = if_norm_reg
        self.z_dim = z_dim

        # loss
        if distance == "l2":
            print("Use distance: l2")
            self.loss_lpips_fn = lambda x, y: 0.0
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])

        elif distance == "l2-lpips":
            print("Use distance: lpips + l2")
            self.loss_lpips_fn = lambda x, y: self.lpips_model.forward(
                x, y, normalize=False
            ).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])

    def forward(self, z, x_gt):
        self.x_hat = self.diffusion_wrapper(self.model)
        self.loss_lpips = self.loss_lpips_fn(self.x_hat, x_gt)
        self.loss_l2 = self.loss_l2_fn(self.x_hat, x_gt)
        self.vec_loss = LAMBDA2 * self.loss_lpips + self.loss_l2

        if self.if_norm_reg:
            z_ = z.view(-1, self.z_dim)
            norm = torch.sum(z_ ** 2, dim=1)
            norm_penalty = (norm - self.z_dim) ** 2
            self.vec_loss += LAMBDA3 * norm_penalty

        return self.vec_loss

    def diffusion_wrapper(self, model, xT=None):
        if xT is None:
            image_size = IMAGE_METADATA.image_size
            xT = (
                torch.randn(
                    BATCH_SIZE, IMAGE_METADATA.num_channels, image_size, image_size
                )
                    .float()
                    .to(DEVICE)
            )
        gen_image = DIFFUSION.sample_from_reverse_process(
            model, xT, SAMPLING_STEPS, {"y": None}, ddim=True
        )
        return gen_image


def optimize_z_bb(loss_model, init_val, query_imgs, save_dir, max_func):
    ### store results
    all_loss = []
    all_z = []
    all_x_hat = []

    ### run the optimization for all query data
    size = len(query_imgs)
    for i in tqdm(range(size // BATCH_SIZE)):
        # for i in tqdm(range(size // (2*BATCH_SIZE),size // BATCH_SIZE)):
        save_dir_batch = os.path.join(save_dir, str(i))

        try:
            x_batch = query_imgs[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            x_gt = torch.from_numpy(x_batch).permute(0, 3, 1, 2).cuda()

            if os.path.exists(save_dir_batch):
                pass
            else:
                visualize_gt(x_batch, check_folder(save_dir_batch))

                ### optimize
                loss_progress = []

                def objective(z):
                    z_ = torch.from_numpy(z).float().view(1, -1, 1, 1).cuda()
                    vec_loss = loss_model.forward(z_, x_gt)
                    vec_loss_np = vec_loss.detach().cpu().numpy()
                    loss_progress.append(vec_loss_np)
                    final_loss = torch.mean(vec_loss)
                    return final_loss.detach().cpu().numpy()

                z0 = init_val[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                options = {"maxiter": max_func, "disp": 1}
                res = minimize(objective, z0, method="Powell", options=options)
                z_curr = res.x
                vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                x_hat_curr = loss_model.x_hat.data.cpu().numpy()
                x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])

                loss_lpips = loss_model.loss_lpips.data.cpu().numpy() if DISTANCE == "l2-lpips" else 0
                loss_l2 = loss_model.loss_l2.data.cpu().numpy()
                save_files(save_dir_batch, ["l2", "lpips"], [loss_l2, loss_lpips])

                ### store results
                visualize_progress(
                    x_hat_curr, vec_loss_curr, save_dir_batch, len(loss_progress)
                )  # visualize finale
                all_loss.append(vec_loss_curr)
                all_z.append(z_curr)
                all_x_hat.append(x_hat_curr)

                save_files(
                    save_dir_batch,
                    ["full_loss", "z", "xhat", "loss_progress"],
                    [vec_loss_curr, z_curr, x_hat_curr, np.array(loss_progress)],
                )

        except KeyboardInterrupt:
            print("Stop optimization\n")
            break
    try:
        all_loss = np.concatenate(all_loss)
        all_z = np.concatenate(all_z)
        all_x_hat = np.concatenate(all_x_hat)
    except:
        all_loss = np.array(all_loss)
        all_z = np.array(all_z)
        all_x_hat = np.array(all_x_hat)
    return all_loss, all_z, all_x_hat


#############################################################################################################
# main
#############################################################################################################
def main():
    args, save_dir, load_dir = check_args(parse_arguments())

    # Variables
    global BATCH_SIZE
    global DEVICE
    global IMAGE_METADATA
    global SAMPLING_STEPS
    global DIFFUSION
    global USE_32x32_GRAYSCALE
    global DISTANCE
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    IMAGE_METADATA = EasyDict(
        {
            "image_size": 32,
            "num_classes": 4,
            "train_images": 109036,
            "val_images": 12376,
            "num_channels": 1,
        }
    )
    SAMPLING_STEPS = args.sampling_steps
    DIFFUSION = GuassianDiffusion(args.diffusion_steps, args.device)
    USE_32x32_GRAYSCALE = True
    DISTANCE = args.distance

    # Set up Model
    # --------------- Create Model --------------
    arch = "UNet"
    model = unets.__dict__[arch](
        image_size=IMAGE_METADATA.image_size,
        in_channels=IMAGE_METADATA.num_channels,
        out_channels=IMAGE_METADATA.num_channels,
        num_classes=None,
    ).to(args.device)
    # --------------- Load Model ----------------
    pretrained_ckpt = load_dir
    print(f"Loading pretrained model from {pretrained_ckpt}")
    d = fix_legacy_dict(torch.load(pretrained_ckpt, map_location=args.device))
    dm = model.state_dict()
    model.load_state_dict(d, strict=False)
    print(
        f"Mismatched keys in ckpt and model: ",
        set(d.keys()) ^ set(dm.keys()),
    )
    print(f"Loaded pretrained model from {pretrained_ckpt}")
    # -------------------------------------------

    Z_DIM = model.in_channels * (model.image_size ** 2)
    resolution = args.resolution

    # Define loss
    loss_model = Loss(model, args.distance, if_norm_reg=False, z_dim=Z_DIM)

    # Initialization
    if args.initialize_type == "zero":
        init_val = np.zeros((args.data_num, Z_DIM, 1, 1))
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == "random":
        np.random.seed(RANDOM_SEED)
        init_val_np = np.random.normal(size=(Z_DIM, 1, 1))
        init_val_np = init_val_np / np.sqrt(np.mean(np.square(init_val_np)) + 1e-8)
        init_val = np.tile(init_val_np, (args.data_num, 1, 1, 1)).astype(np.float32)
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == "nn":
        idx = 0
        init_val_pos = np.load(os.path.join(args.nn_dir, "pos_z.npy"))[:, idx, :]
        init_val_pos = np.reshape(init_val_pos, [len(init_val_pos), Z_DIM, 1, 1])
        init_val_neg = np.load(os.path.join(args.nn_dir, "neg_z.npy"))[:, idx, :]
        init_val_neg = np.reshape(init_val_neg, [len(init_val_neg), Z_DIM, 1, 1])
    else:
        raise NotImplementedError

    # Positive
    pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext="jpg")[
                     : args.data_num
                     ]
    pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])
    if USE_32x32_GRAYSCALE:
        pos_query_imgs = [Image.open(f) for f in pos_data_paths]
        transform_train = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
            ]
        )
        pos_query_imgs = np.array([transform_train(f).permute(2, 1, 0).numpy() for f in pos_query_imgs])

    query_loss, query_z, query_xhat = optimize_z_bb(
        loss_model,
        init_val_pos,
        pos_query_imgs,
        check_folder(os.path.join(save_dir, "pos_results")),
        args.maxfunc,
    )
    save_results(save_dir, "pos", query_loss, query_z, query_xhat)

    # Negative
    neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext="jpg")[
                     : args.data_num
                     ]
    neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])
    if USE_32x32_GRAYSCALE:
        neg_query_imgs = [Image.open(f) for f in neg_data_paths]
        transform_train = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
            ]
        )
        neg_query_imgs = np.array([transform_train(f).permute(2, 1, 0).numpy() for f in neg_query_imgs])

    query_loss, query_z, query_xhat = optimize_z_bb(
        loss_model,
        init_val_neg,
        neg_query_imgs,
        check_folder(os.path.join(save_dir, "neg_results")),
        args.maxfunc,
    )
    save_results(save_dir, "neg", query_loss, query_z, query_xhat)


def save_results(save_dir, sign, query_loss, query_z, query_xhat):
    save_files(save_dir, [f"{sign}_loss"], [query_loss])
    save_files(save_dir, [f"{sign}_z"], [query_z])
    save_files(save_dir, [f"{sign}_xhat"], [query_xhat])


if __name__ == "__main__":
    main()
