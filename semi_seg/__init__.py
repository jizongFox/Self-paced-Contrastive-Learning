acdc_ratios = [1, 174]
prostate_ratio = [3, 40]  # 2 4 8
prostate_md_ratio = [1, 17]
mmwhsct_ratio = [1, 10]
mmwhsmr_ratio = [1, 10]
if True:
    acdc_ratios = [1, 2, 4, 174]
    prostate_ratio = [3, 5, 7, 40]  # 2, 4, 8, 40
    mmwhsct_ratio = [1, 2, 10]
    prostate_md_ratio = [1, 2, 4, 17]  # 1, 2, 4, 8
    mmwhsmr_ratio = [1, 2, 10]

pre_max_epoch_zoo = {
    "acdc": 80,
    "mmwhsct": 80,
    "mmwhsmr": 80,
    "prostate": 80,
}
ft_max_epoch_zoo = {
    "acdc": 60,
    "mmwhsct": 60,
    "mmwhsmr": 60,
    "prostate": 80,
}
num_batches_zoo = {
    "acdc": 200,
    "mmwhsct": 350,
    "mmwhsmr": 350,
    "prostate": 300,
}

ratio_zoo = {
    "acdc": acdc_ratios,
    "prostate": prostate_ratio,
    "prostate_md": prostate_md_ratio,
    "mmwhsct": mmwhsct_ratio,
    "mmwhsmr": mmwhsmr_ratio,
}
data2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "prostate_md": 3,
    "spleen": 2,
    "mmwhsct": 5,
    "mmwhsmr": 5,

}
data2input_dim = {
    "acdc": 1,
    "prostate": 1,
    "prostate_md": 1,
    "spleen": 1,
    "mmwhsct": 1,
    "mmwhsmr": 1,
}

pre_lr_zooms = {
    "acdc": 0.0000005,
    "prostate": 0.0000005,
    "prostate_md": 0.000005,
    "mmwhsct": 0.0000005,
    "mmwhsmr": 0.0000005
}

ft_lr_zooms = {
    "acdc": 0.0000002,
    "prostate": 0.0000005,
    "prostate_md": 0.0000005,
    "spleen": 0.000001,
    "mmwhsct": 0.000002,
    "mmwhsmr": 0.000002
}
__accounts = ["rrg-mpederso", ]
# __accounts = ["def-chdesa", "rrg-mpederso", "def-mpederso"]

labeled_filenames = {
    "acdc": {1: ["patient100_00"],
             2: ["patient027_01", "patient100_00"],
             4: ["patient027_01", "patient038_01", "patient067_01", "patient100_00"],
             8: ["patient027_01", "patient038_01", "patient067_01", "patient100_00", "patient002_00", "patient004_00",
                 "patient006_01", "patient007_00"]
             },
    "prostate": {3: ["Case10", "Case17", "Case45"],
                 5: ["Case00", "Case10", "Case17", "Case37", "Case45"],
                 7: ["Case00", "Case10", "Case17", "Case34", "Case37", "Case38", "Case45"]},
    "mmwhsct": {1: ["1003"],
                2: ["1003", "1010"]}
}
