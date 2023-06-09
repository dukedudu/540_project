# Adapted based on https://github.com/yaoxufeng/PCL-Proxy-based-Contrastive-Learning-for-Domain-Generalization
# and https://github.com/facebookresearch/DomainBed
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

import pandas as pd
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import seaborn as sns

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def accuracy_from_loader(algorithm, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():

            _, logits, _ = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset: weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float()
                        * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float()
                        * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss


def tsne_acc_from_loader(algorithm, loader, weights, env, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()
    start_test = True
    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():

            rep, logits, _ = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset: weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float()
                        * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float()
                        * batch_weights).sum().item()

        total += batch_weights.sum().item()
        batch_weights = batch_weights.type(torch.int)
        rep = torch.repeat_interleave(rep, batch_weights, dim=0)
        y = torch.repeat_interleave(y, batch_weights, dim=0)

        if start_test:
            all_fea = rep.float().cpu()
            all_y = y.float().cpu()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, rep.float().cpu()), 0)
            all_y = torch.cat((all_y, y.float().cpu()), 0)

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    df_subset = pd.DataFrame()

    tsne = TSNE(n_components=2, verbose=0, perplexity=2, n_iter=300)
    tsne_results = tsne.fit_transform(all_fea)

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['y'] = all_y

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", len(torch.unique(all_y))),
        data=df_subset,
        legend=False,
        # legend="full",
        # alpha=0.5
    )
    fig_name = 'visualization/' + \
        str(env) + 'tsne_e' + '-' + str(acc)[:5] + '.png'
    plt.savefig(fig_name)
    plt.close()
    return acc, loss


def accuracy(algorithm, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, **kwargs)


def accuracy_tsne(algorithm, loader_kwargs, weights, env_name, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return tsne_acc_from_loader(algorithm, loader, weights, env_name, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            if is_test:
                acc, loss = accuracy_tsne(algorithm, loader_kwargs,
                                          weights, env_name, debug=self.debug)
            else:
                acc, loss = accuracy(algorithm, loader_kwargs,
                                     weights, debug=self.debug)

            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
