import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import sys


from torch.utils.data import DataLoader
from src.logger.logger import _logger, _configLogger
from src.dataset.dataset import SimpleIterDataset
from src.utils.import_tools import import_module
from src.layers.batch_operations import graph_batch_func

def set_gpus(args):
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
        dev = torch.device(gpus[0])
        print("Using GPUs:", gpus)
    else:
        print("No GPUs flag provided - Setting GPUs to [0]")
        gpus = [0]
        dev = torch.device(gpus[0])
        raise Exception("Please provide GPU number")
    return gpus, dev

    
def get_gpu_dev(args):
    if args.gpus != "":
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = 0
        devices = 0
    return accelerator, devices


def model_setup(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name="_network_module")
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}

    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]  # ?
        dev = torch.device(gpus[0])
    else:
        gpus = None
        local_rank = 0
        dev = torch.device("cpu")
    model, model_info = network_module.get_model(
        data_config, args=args, dev=dev, **network_options
    )
    return model.mod


def get_samples_steps_per_epoch(args):
    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!"
            )
    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!"
            )
    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(
            args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split
        )
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None
    return args


def to_filelist(args, mode="train"):
    if mode == "train":
        flist = args.data_train
    elif mode == "val":
        flist = args.data_val
    else:
        raise NotImplementedError("Invalid mode %s" % mode)

    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    for f in flist:
        if ":" in f:
            name, fp = f.split(":")
        else:
            name, fp = "_", f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    if args.local_rank is not None:
        if mode == "train":
            gpus_list, _ = set_gpus(args)
            local_world_size = len(gpus_list)  # int(os.environ['LOCAL_WORLD_SIZE'])
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[args.local_rank :: local_world_size]
                assert len(new_files) > 0
                np.random.shuffle(new_files)
                new_file_dict[name] = new_files
            file_dict = new_file_dict

    if args.copy_inputs:
        import tempfile

        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        new_file_dict = {name: [] for name in file_dict}
        for name, files in file_dict.items():
            for src in files:
                dest = os.path.join(tmpdir, src.lstrip("/"))
                if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                _logger.info("Copied file %s to %s" % (src, dest))
                new_file_dict[name].append(dest)
            if len(files) != len(new_file_dict[name]):
                _logger.error(
                    "Only %d/%d files copied for %s file group %s",
                    len(new_file_dict[name]),
                    len(files),
                    mode,
                    name,
                )
        file_dict = new_file_dict

    filelist = sum(file_dict.values(), [])
    assert len(filelist) == len(set(filelist))
    # print(args.local_rank, len(filelist))
    return file_dict, filelist


def train_load(args):
    """
    Loads the training data.
    :param args:
    :return: train_loader, val_loader, data_config, train_inputs
    """
    train_file_dict, train_files = to_filelist(args, "train")
    if args.data_val:
        val_file_dict, val_files = to_filelist(args, "val")
        train_range = val_range = (0, 1)
    else:
        val_file_dict, val_files = train_file_dict, train_files
        train_range = (0, args.train_val_split)
        val_range = (args.train_val_split, 1)
    _logger.info(
        "Using %d files for training, range: %s" % (len(train_files), str(train_range))
    )
    _logger.info(
        "Using %d files for validation, range: %s" % (len(val_files), str(val_range))
    )

    if args.demo:
        train_files = train_files[:20]
        val_files = val_files[:20]
        train_file_dict = {"_": train_files}
        val_file_dict = {"_": val_files}
        _logger.info(train_files)
        _logger.info(val_files)
        args.data_fraction = 0.1
        args.fetch_step = 0.002

    if args.in_memory and (
        args.steps_per_epoch is None or args.steps_per_epoch_val is None
    ):
        raise RuntimeError("Must set --steps-per-epoch when using --in-memory!")
    syn_str = args.synthetic_graph_npart_range
    synthetic = syn_str != ""
    minp, maxp = (
        0,
        0,
    )
    if synthetic:
        minp = int(syn_str.split("-")[0])
        maxp = int(syn_str.split("-")[1])

    train_data = SimpleIterDataset(
        train_file_dict,
        args.data_config,
        for_training=True,
        extra_selection=args.extra_selection,
        remake_weights=not args.no_remake_weights,
        load_range_and_fraction=(train_range, args.data_fraction),
        file_fraction=args.file_fraction,
        fetch_by_files=args.fetch_by_files,
        fetch_step=args.fetch_step,
        infinity_mode=args.steps_per_epoch is not None,
        in_memory=args.in_memory,
        laplace=args.laplace,
        diffs=args.diffs,
        edges=args.class_edges,
        name="train" + ("" if args.local_rank is None else "_rank%d" % args.local_rank),
        dataset_cap=args.train_cap,
        n_noise=args.n_noise,
        synthetic=synthetic,
        synthetic_npart_min=minp,
        synthetic_npart_max=maxp,
    )
    val_data = SimpleIterDataset(
        val_file_dict,
        args.data_config,
        for_training=True,
        extra_selection=args.extra_selection,
        load_range_and_fraction=(val_range, args.data_fraction),
        file_fraction=args.file_fraction,
        fetch_by_files=args.fetch_by_files,
        fetch_step=args.fetch_step,
        infinity_mode=args.steps_per_epoch_val is not None,
        in_memory=args.in_memory,
        laplace=args.laplace,
        diffs=args.diffs,
        edges=args.class_edges,
        name="val" + ("" if args.local_rank is None else "_rank%d" % args.local_rank),
        dataset_cap=args.val_cap,
        n_noise=args.n_noise,
        synthetic=synthetic,
        synthetic_npart_min=minp,
        synthetic_npart_max=maxp,
    )

    if args.class_edges:
        collator_func = graph_batch_func_edges
    else:
        collator_func = graph_batch_func
    # train_data_arg = train_data
    # val_data_arg = val_data
    # if args.train_cap == 1:
    #    train_data_arg = [next(iter(train_data_arg))]
    # if args.val_cap == 1:
    #    val_data_arg = [next(iter(val_data_arg))]
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
        collate_fn=collator_func,
        persistent_workers=args.num_workers > 0 and args.steps_per_epoch is not None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        collate_fn=collator_func,
        num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
        persistent_workers=args.num_workers > 0
        and args.steps_per_epoch_val is not None,
    )

    data_config = train_data.config
    train_input_names = train_data.config.input_names
    train_label_names = 0  # train_data.config.label_names

    return train_loader, val_loader, data_config, train_input_names


def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ":" in f:
            name, fp = f.split(":")
            if "%" in name:
                name, split = name.split("%")
                split_dict[name] = int(split)
        else:
            name, fp = "", f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f"{name}_{i}"] = files[i * split : (i + 1) * split]
    print("file_dict", file_dict)

    def get_test_loader(name):
        filelist = file_dict[name]
        _logger.info(
            "Running on test file group %s with %d files:\n...%s",
            name,
            len(filelist),
            "\n...".join(filelist),
        )
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset(
            {name: filelist},
            args.data_config,
            for_training=False,
            extra_selection=args.extra_test_selection,
            load_range_and_fraction=((0, 1), args.data_fraction),
            fetch_by_files=True,
            fetch_step=1,
            name="test_" + name,
        )
        test_loader = DataLoader(
            test_data,
            num_workers=num_workers,
            batch_size=args.batch_size,
            drop_last=False,
            pin_memory=True,
            collate_fn=graph_batch_func,
        )
        return test_loader

    test_loaders = {
        name: functools.partial(get_test_loader, name) for name in file_dict
    }
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config


def onnx(args):
    """
    Saving model as ONNX.
    :param args:
    :return:
    """
    assert args.export_onnx.endswith(".onnx")
    model_path = args.model_prefix
    _logger.info("Exporting model %s to ONNX" % model_path)

    from src.dataset.dataset import DataConfig

    data_config = DataConfig.load(
        args.data_config, load_observers=False, load_reweight_info=False
    )
    model, model_info, _ = model_setup(args, data_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.cpu()
    model.eval()

    os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
    inputs = tuple(
        torch.ones(model_info["input_shapes"][k], dtype=torch.float32)
        for k in model_info["input_names"]
    )
    torch.onnx.export(
        model,
        inputs,
        args.export_onnx,
        input_names=model_info["input_names"],
        output_names=model_info["output_names"],
        dynamic_axes=model_info.get("dynamic_axes", None),
        opset_version=13,
    )
    _logger.info("ONNX model saved to %s", args.export_onnx)

    preprocessing_json = os.path.join(
        os.path.dirname(args.export_onnx), "preprocess.json"
    )
    data_config.export_json(preprocessing_json)
    _logger.info("Preprocessing parameters saved to %s", preprocessing_json)


def flops(model, model_info):
    """
    Count FLOPs and params.
    :param args:
    :param model:
    :param model_info:
    :return:
    """
    from src.utils.utils.flops_counter import get_model_complexity_info
    import copy

    model = copy.deepcopy(model).cpu()
    model.eval()

    inputs = tuple(
        torch.ones(model_info["input_shapes"][k], dtype=torch.float32)
        for k in model_info["input_names"]
    )

    macs, params = get_model_complexity_info(
        model, inputs, as_strings=True, print_per_layer_stat=True, verbose=True
    )
    _logger.info("{:<30}  {:<8}".format("Computational complexity: ", macs))
    _logger.info("{:<30}  {:<8}".format("Number of parameters: ", params))


def profile(args, model, model_info, device):
    """
    Profile.
    :param model:
    :param model_info:
    :return:
    """
    import copy
    from torch.profiler import profile, record_function, ProfilerActivity

    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()

    inputs = tuple(
        torch.ones(
            (args.batch_size,) + model_info["input_shapes"][k][1:], dtype=torch.float32
        ).to(device)
        for k in model_info["input_names"]
    )
    for x in inputs:
        print(x.shape, x.device)

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=2),
        on_trace_ready=trace_handler,
    ) as p:
        for idx in range(100):
            model(*inputs)
            p.step()


def optim(args, model, device):
    """
    Optimizer and scheduler.
    :param args:
    :param model:
    :return:
    """
    optimizer_options = {k: ast.literal_eval(v) for k, v in args.optimizer_option}
    _logger.info("Optimizer options: %s" % str(optimizer_options))

    names_lr_mult = []
    if "weight_decay" in optimizer_options or "lr_mult" in optimizer_options:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py#L31
        import re

        decay, no_decay = {}, {}
        names_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (
                len(param.shape) == 1
                or name.endswith(".bias")
                or (
                    hasattr(model, "no_weight_decay")
                    and name in model.no_weight_decay()
                )
            ):
                no_decay[name] = param
                names_no_decay.append(name)
            else:
                decay[name] = param

        decay_1x, no_decay_1x = [], []
        decay_mult, no_decay_mult = [], []
        mult_factor = 1
        if "lr_mult" in optimizer_options:
            pattern, mult_factor = optimizer_options.pop("lr_mult")
            for name, param in decay.items():
                if re.match(pattern, name):
                    decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    decay_1x.append(param)
            for name, param in no_decay.items():
                if re.match(pattern, name):
                    no_decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    no_decay_1x.append(param)
            assert len(decay_1x) + len(decay_mult) == len(decay)
            assert len(no_decay_1x) + len(no_decay_mult) == len(no_decay)
        else:
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
        wd = optimizer_options.pop("weight_decay", 0.0)
        parameters = [
            {"params": no_decay_1x, "weight_decay": 0.0},
            {"params": decay_1x, "weight_decay": wd},
            {
                "params": no_decay_mult,
                "weight_decay": 0.0,
                "lr": args.start_lr * mult_factor,
            },
            {
                "params": decay_mult,
                "weight_decay": wd,
                "lr": args.start_lr * mult_factor,
            },
        ]
        _logger.info(
            "Parameters excluded from weight decay:\n - %s",
            "\n - ".join(names_no_decay),
        )
        if len(names_lr_mult):
            _logger.info(
                "Parameters with lr multiplied by %s:\n - %s",
                mult_factor,
                "\n - ".join(names_lr_mult),
            )
    else:
        parameters = model.parameters()

    if args.optimizer == "ranger":
        from src.utils.nn.optimizer.ranger import Ranger

        opt = Ranger(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == "adamW":
        opt = torch.optim.AdamW(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == "radam":
        opt = torch.optim.RAdam(parameters, lr=args.start_lr, **optimizer_options)

    # load previous training and resume if `--load-epoch` is set
    if args.load_epoch is not None:
        _logger.info("Resume training from epoch %d" % args.load_epoch)
        model_state = torch.load(
            args.model_prefix + "_epoch-%d_state.pt" % args.load_epoch,
            map_location=device,
        )
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        opt_state_file = args.model_prefix + "_epoch-%d_optimizer.pt" % args.load_epoch
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            opt.load_state_dict(opt_state)
        else:
            _logger.warning("Optimizer state file %s NOT found!" % opt_state_file)

    scheduler = None
    if args.lr_finder is None:
        if args.lr_scheduler == "steps":
            lr_step = round(args.num_epochs / 3)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=[10],  # [lr_step, 2 * lr_step],
                gamma=0.20,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch,
            )
        elif args.lr_scheduler == "flat+decay":
            num_decay_epochs = max(1, int(args.num_epochs * 0.3))
            milestones = list(
                range(args.num_epochs - num_decay_epochs, args.num_epochs)
            )
            gamma = 0.01 ** (1.0 / num_decay_epochs)
            if len(names_lr_mult):

                def get_lr(epoch):
                    return gamma ** max(0, epoch - milestones[0] + 1)  # noqa

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt,
                    (lambda _: 1, lambda _: 1, get_lr, get_lr),
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch,
                    verbose=True,
                )
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt,
                    milestones=milestones,
                    gamma=gamma,
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch,
                )
        elif args.lr_scheduler == "flat+linear" or args.lr_scheduler == "flat+cos":
            total_steps = args.num_epochs * args.steps_per_epoch
            warmup_steps = args.warmup_steps
            flat_steps = total_steps * 0.7 - 1
            min_factor = 0.001

            def lr_fn(step_num):
                if step_num > total_steps:
                    raise ValueError(
                        "Tried to step {} times. The specified number of total steps is {}".format(
                            step_num + 1, total_steps
                        )
                    )
                if step_num < warmup_steps:
                    return 1.0 * step_num / warmup_steps
                if step_num <= flat_steps:
                    return 1.0
                pct = (step_num - flat_steps) / (total_steps - flat_steps)
                if args.lr_scheduler == "flat+linear":
                    return max(min_factor, 1 - pct)
                else:
                    return max(min_factor, 0.5 * (math.cos(math.pi * pct) + 1))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_fn,
                last_epoch=-1
                if args.load_epoch is None
                else args.load_epoch * args.steps_per_epoch,
            )
            scheduler._update_per_step = (
                True  # mark it to update the lr every step, instead of every epoch
            )
        elif args.lr_scheduler == "one-cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=args.start_lr,
                epochs=args.num_epochs,
                steps_per_epoch=args.steps_per_epoch,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch,
            )
            scheduler._update_per_step = (
                True  # mark it to update the lr every step, instead of every epoch
            )
        elif args.lr_scheduler == "reduceplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=2, threshold=0.01
            )
            # scheduler._update_per_step = (
            #     True  # mark it to update the lr every step, instead of every epoch
            # )
            scheduler._update_per_step = (
                False  # mark it to update the lr every step, instead of every epoch
            )
    return opt, scheduler


def iotest(args, data_loader):
    """
    Io test
    :param args:
    :param data_loader:
    :return:
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    from src.data.tools import _concat

    _logger.info("Start running IO test")
    monitor_info = defaultdict(list)

    for X, y, Z in tqdm(data_loader):
        for k, v in Z.items():
            monitor_info[k].append(v.cpu().numpy())
    monitor_info = {k: _concat(v) for k, v in monitor_info.items()}
    if monitor_info:
        monitor_output_path = "weaver_monitor_info.pkl"
        import pickle

        with open(monitor_output_path, "wb") as f:
            pickle.dump(monitor_info, f)
        _logger.info("Monitor info written to %s" % monitor_output_path)


def save_root(args, output_path, data_config, scores, labels, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    from src.data.fileio import _write_root

    output = {}
    if args.regression_mode:
        output[data_config.label_names[0]] = labels[data_config.label_names[0]]
        output["output"] = scores
    else:
        for idx, label_name in enumerate(data_config.label_value):
            output[label_name] = labels[data_config.label_names[0]] == idx
            output["score_" + label_name] = scores[:, idx]
    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        if v.ndim > 1:
            _logger.warning("Ignoring %s, not a 1d array.", k)
            continue
        output[k] = v
    for k, v in observers.items():
        if v.ndim > 1:
            _logger.warning("Ignoring %s, not a 1d array.", k)
            continue
        output[k] = v
    _write_root(output_path, output)


def save_parquet(args, output_path, scores, labels, observers):
    """
    Saves as parquet file
    :param scores:
    :param labels:
    :param observers:
    :return:
    """
    import awkward as ak

    output = {"scores": scores}
    output.update(labels)
    output.update(observers)
    ak.to_parquet(ak.Array(output), output_path, compression="LZ4", compression_level=4)


def count_parameters(model):
    return sum(p.numel() for p in model.mod.parameters() if p.requires_grad)


def model_setup1(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config1, name="_network_module")
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    network_options.update(data_config.custom_model_kwargs)
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]  # ?
        dev = torch.device(gpus[0])
    else:
        gpus = None
        local_rank = 0
        dev = torch.device("cpu")
    model, model_info = network_module.get_model(
        data_config, args=args, dev=dev, **network_options
    )

    if args.load_model_weights_1:
        print("Loading model state from %s" % args.load_model_weights_1)
        model_state = torch.load(args.load_model_weights_1, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info(
            "Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s"
            % (args.load_model_weights, missing_keys, unexpected_keys)
        )

    loss_func = torch.nn.CrossEntropyLoss()

    return model, model_info, loss_func
