import argparse

from deepclustering2.cchelper import JobSubmiter
from deepclustering2.utils import gethash

from semi_seg import data2class_numbers, ft_lr_zooms, data2input_dim
from semi_seg.scripts.helper import BindPretrainFinetune, BindContrastive, \
    BindSelfPaced, run_jobs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

comm_parser = parser.add_argument_group("common options")

comm_parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
comm_parser.add_argument("-b", "--num_batches", default=200, type=int, help="num batches")
comm_parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
comm_parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
comm_parser.add_argument("--on-local", default=False, action="store_true", help="run on local")
comm_parser.add_argument("--time", type=int, default=4, help="submitted time to CC")
comm_parser.add_argument("--show_cmd", default=False, action="store_true", help="only show generated cmd.")

subparser = parser.add_subparsers(dest='stage')
baseline = subparser.add_parser("baseline")
infonce = subparser.add_parser("infonce")
selfpaced = subparser.add_parser("selfpaced")

# baseline
baseline.add_argument("-e", "--max_epoch", type=str, default=75, help="max_epoch")
baseline.add_argument("--lr", type=str, default=None, help="learning rate")

# infonce
BindPretrainFinetune.bind(infonce)
BindContrastive.bind(infonce)

# self-paced
BindPretrainFinetune.bind(selfpaced)
BindContrastive.bind(selfpaced)
BindSelfPaced.bind(selfpaced)

args = parser.parse_args()

# setting common params
__git_hash__ = gethash(__file__)
dataset_name = args.dataset_name
num_batches = args.num_batches
random_seed = args.random_seed
num_classes = data2class_numbers[args.dataset_name]
input_dim = data2input_dim[args.dataset_name]
save_dir = args.save_dir

save_dir += ("/" + "/".join(
    [
        f"githash_{__git_hash__[:7]}",
        args.dataset_name,
        f"random_seed_{random_seed}"
    ]))

SharedParams = f" Data.name={dataset_name}" \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={num_classes} " \
               f" Arch.input_dim={input_dim} " \
               f" RandomSeed={random_seed} "

if args.stage == "baseline":
    max_epoch = args.max_epoch
    lr = args.lr or f"{ft_lr_zooms[args.dataset_name]:.10f}"
    job_array = [
        f"python main_finetune.py {SharedParams} Optim.lr={lr} Trainer.max_epoch={max_epoch} "
        f"Trainer.name=finetune Trainer.save_dir={save_dir}/baseline "
    ]


elif args.stage == "infonce":
    parser1 = BindPretrainFinetune()
    parser1.parse(args)
    parser2 = BindContrastive()
    parser2.parse(args)

    group_sample_num = args.group_sample_num
    gfeature_names = args.global_features
    gimportance = args.global_importance
    contrast_on = args.contrast_on
    save_dir += f"/sample_num_{group_sample_num}"

    subpath = f"global_{'_'.join([*gfeature_names, *[str(x) for x in gimportance]])}/" \
              f"contrast_on_{'_'.join(contrast_on)}"

    string = f"python main_infonce.py Trainer.name=infoncepretrain {SharedParams} " \
             f" {parser1.get_option_str()} " \
             f" {parser2.get_option_str()} " \
             f" Trainer.save_dir={save_dir}/{subpath}/infonce " \
             f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce.yaml"

    job_array = [string]


elif args.stage == "selfpaced":
    parser1 = BindPretrainFinetune()
    parser1.parse(args)
    parser2 = BindContrastive()
    parser2.parse(args)
    parser3 = BindSelfPaced()
    parser3.parse(args)

    group_sample_num = args.group_sample_num
    gfeature_names = args.global_features
    gimportance = args.global_importance
    contrast_on = args.contrast_on
    assert len(gfeature_names) == len(contrast_on)
    begin_value, end_value = args.begin_value, args.end_value
    method = args.method
    type = args.scheduler_type

    save_dir += f"/sample_num_{group_sample_num}/"

    subpath = f"global_{'_'.join([*gfeature_names, *[str(x) for x in gimportance]])}/" \
              f"contrast_on_{'_'.join(contrast_on)}"


    def get_loss_str(begin_value, end_value):

        _tmp = [f"{b}_{e}" for b, e in zip(begin_value, end_value)]
        return "loss_params*" + "*".join(_tmp)


    string = f"python main_infonce.py Trainer.name=infoncepretrain" \
             f" {SharedParams} {parser1.get_option_str()} " \
             f" {parser2.get_option_str()} " \
             f" {parser3.get_option_str()} " \
             f" Trainer.save_dir={save_dir}/{subpath}/self-paced/method_{'_'.join(method)}/" \
             f"{get_loss_str(begin_value, end_value)}/type_{'_'.join(type)} " \
             f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/selfpaced_infonce.yaml"
    job_array = [string]

else:
    raise NotImplementedError(args.stage)

job_submiter = JobSubmiter(project_path="../", on_local=args.on_local, time=args.time, mem=64)

run_jobs(job_submiter, job_array, args, )
