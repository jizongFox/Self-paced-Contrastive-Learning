# this script gives the checkpoint and perform semi-supervised learning accordingly.
# involving 1 patient, 2 patient, 4 patient for ACDC dataset.
import argparse

from deepclustering2.cchelper import JobSubmiter
from deepclustering2.utils import load_yaml

from semi_seg import ratio_zoo, data2class_numbers, ft_lr_zooms, data2input_dim
from semi_seg.scripts.helper import __git_hash__, dump_config, \
    run_jobs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

comm_parser = parser.add_argument_group("common options")

comm_parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
comm_parser.add_argument("-b", "--num_batches", default=200, type=int, help="num batches")
comm_parser.add_argument("-e", "--max_epoch", default=80, type=int, help="num batches")
comm_parser.add_argument("-s", "--random_seed", default=10, type=int, help="random seed")
comm_parser.add_argument("--lr", default=None, type=float, help="learning rate for semi supervised learning")
comm_parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
comm_parser.add_argument("--on-local", default=False, action="store_true", help="run on local")
comm_parser.add_argument("--time", type=int, default=4, help="submitted time to CC")
comm_parser.add_argument("--show_cmd", default=False, action="store_true", help="only show generated cmd.")
comm_parser.add_argument("--arch_checkpoint", default="null", type=str, help="network checkpoint")

subparser = parser.add_subparsers(dest='stage')
baseline = subparser.add_parser("baseline")
meanteacher = subparser.add_parser("meanteacher")
infonce = subparser.add_parser("infonce")
meanteacherinfonce = subparser.add_parser("meanteacherinfonce")
udaiic = subparser.add_parser("udaiic")
entropy = subparser.add_parser("entropy")

# mean teacher
meanteacher.add_argument("--mt_weight", default=1e-04, type=float, help="mean teacher weight coefficient")

# infonce
infonce.add_argument("--config_path", required=True, help="configuration for contrastive learning")
infonce.add_argument("--info_weight", default=1e-04, type=float, help="infonce weight coefficient")

# mean teacher + infonce
meanteacherinfonce.add_argument("--mt_weight", default=1e-04, type=float, help="mean teacher weight coefficient")
meanteacherinfonce.add_argument("--config_path", required=True, help="configuration for contrastive learning")
meanteacherinfonce.add_argument("--info_weight", default=1e-04, type=float, help="infonce weight coefficient")

# udaiic
udaiic.add_argument("--uda_weight", default=0.1, type=str, help="uda weight")
udaiic.add_argument("--iic_weight", default=0.1, type=str, help="iic weight")

entropy.add_argument("--ent_weight", default=0.01, type=str, help="entropy minimization weight")

args = parser.parse_args()

labeled_ratios = ratio_zoo[args.dataset_name]

if 1.0 in labeled_ratios:
    labeled_ratios.remove(1.0)

# setting common params
dataset_name = args.dataset_name
num_batches = args.num_batches
random_seed = args.random_seed
num_classes = data2class_numbers[args.dataset_name]
input_dim = data2input_dim[args.dataset_name]
lr = args.lr or f"{ft_lr_zooms[args.dataset_name]:.10f}"

save_dir = args.save_dir

save_dir += ("/" + "/".join(
    [
        f"githash_{__git_hash__[:7]}",
        args.dataset_name,
        f"random_seed_{random_seed}",
        f"checkpoint_{'yes' if args.arch_checkpoint != 'null' else 'null'}"
    ]))


def parse_contrastive_args_from_path(config_path):
    config = load_yaml(config_path)
    contrastive_params = config["ProjectorParams"]
    return {"ProjectorParams": contrastive_params}


SharedParams = f" Data.name={dataset_name}" \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={num_classes} " \
               f" Arch.input_dim={input_dim} " \
               f" RandomSeed={random_seed} " \
               f" Trainer.max_epoch={args.max_epoch} " \
               f" Arch.checkpoint={args.arch_checkpoint} " \
               f" Trainer.two_stage_training=true "

if args.stage == "baseline":
    job_array = [
        f"python main.py {SharedParams} Optim.lr={lr}  "
        f"Trainer.name=finetune Trainer.save_dir={save_dir}/baseline/tra/ratio_{str(x)} "
        f"Data.labeled_data_ratio={x}  Data.unlabeled_data_ratio={1 - x} " for x in sorted({*labeled_ratios, 1.0})
    ]
    job_array = [" && ".join(job_array)]

elif args.stage == "infonce":
    info_weight = args.info_weight
    contrastive_params = parse_contrastive_args_from_path(args.config_path)
    with dump_config(contrastive_params) as opt_config_path:
        job_array = [f" python main.py Trainer.name=infonce {SharedParams} Optim.lr={lr} "
                     f" Trainer.save_dir={save_dir}/infonce/weight_{info_weight}/tra/ratio_{str(x)} "
                     f" ProjectorParams.Weight={info_weight:.10f} "
                     f" Data.labeled_data_ratio={x}  "
                     f" Data.unlabeled_data_ratio={1 - x} "
                     f" --opt_config_path {opt_config_path}" for x in labeled_ratios]
    job_array = [" && ".join(job_array)]

elif args.stage == "meanteacher":
    mt_weight = args.mt_weight

    job_array = [
        f"python main.py {SharedParams} Optim.lr={lr}  "
        f" Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/mt_{mt_weight}/tra/ratio_{str(x)} "
        f" MeanTeacherParameters.weight={mt_weight:.10f} "
        f" Data.labeled_data_ratio={x}  Data.unlabeled_data_ratio={1 - x} "
        f" --opt_config_path ../config/specific/mt.yaml" for x in labeled_ratios
    ]

    job_array = [" && ".join(job_array)]

elif args.stage == "meanteacherinfonce":
    mt_weight = args.mt_weight
    info_weight = args.info_weight

    contrastive_params = parse_contrastive_args_from_path(args.config_path)
    with dump_config(contrastive_params) as opt_config_path:
        job_array = [f" python main.py Trainer.name=infoncemt {SharedParams} Optim.lr={lr} "
                     f" Trainer.save_dir={save_dir}/infoncemt/info_{info_weight}_mt_{mt_weight}/tra/ratio_{str(x)} "
                     f" MeanTeacherParameters.weight={mt_weight:.10f} "
                     f" ProjectorParams.Weight={info_weight:.10f} "
                     f" Data.labeled_data_ratio={x}  "
                     f" Data.unlabeled_data_ratio={1 - x} "
                     f" --opt_config_path {opt_config_path} ../config/specific/mt.yaml " for x in labeled_ratios]

    job_array = [" && ".join(job_array)]
elif args.stage == "udaiic":
    uda_weight = args.uda_weight
    iic_weight = args.iic_weight
    job_array = [
        f" python main.py Trainer.name=udaiic {SharedParams} "
        f" IICRegParameters.weight={iic_weight} "
        f" UDARegCriterion.weight={uda_weight} "
        f" Trainer.save_dir={save_dir}/uda_iic/uda_{uda_weight}_iic_{iic_weight}/tra/ratio_{str(x)}"
        f" Data.labeled_data_ratio={x}  "
        f" Data.unlabeled_data_ratio={1 - x} "
        f" --opt_config_path ../config/specific/iic.yaml ../config/specific/uda.yaml" for x in labeled_ratios
    ]
    job_array = [" && ".join(job_array)]
elif args.stage == "entropy":
    ent_weight = args.ent_weight
    job_array = [
        f" python main.py Trainer.name=entropy {SharedParams} "
        f" EntropyMinParameters.weight={ent_weight} "
        f" Trainer.save_dir={save_dir}/entropy/ent_w_{ent_weight}/tra/ratio_{str(x)}"
        f" Data.labeled_data_ratio={x}  "
        f" Data.unlabeled_data_ratio={1 - x} "
        f" --opt_config_path ../config/specific/entmin.yaml " for x in labeled_ratios
    ]
    job_array = [" && ".join(job_array)]
else:
    raise NotImplemented(args.stage)

job_submiter = JobSubmiter(project_path="../", on_local=args.on_local, time=args.time, mem=64)

run_jobs(job_submiter, job_array, args, )
