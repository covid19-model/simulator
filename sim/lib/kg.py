import time
import os
import asyncio
import threading
import json

from lib.priorityqueue import PriorityQueue
from lib.dynamics import DiseaseModel
from lib.mobilitysim import MobilitySimulator
from lib.parallel import *

import gpytorch
import torch
import botorch
import sobol_seq
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, ModelListGP, HeteroskedasticSingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qNoisyExpectedImprovement, qSimpleRegret
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.utils import is_nonnegative
from botorch.utils.sampling import draw_sobol_samples, manual_seed

from botorch.acquisition import OneShotAcquisitionFunction
import botorch.utils.transforms as transforms
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform

from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import GenericMCObjective, ConstrainedMCObjective
from botorch.gen import get_best_candidates, gen_candidates_torch
from torch.quasirandom import SobolEngine
from botorch import settings


import time
import pprint

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

"""
These functions are mostly copy-pasted from the botorch library
https://github.com/pytorch/botorch/tree/master/botorch

and slightly amended in a few places to fix bugs in botorch functions we 
use (and subseqeuntly debug our own calls) that made it impossible to
use botorch out of the box in our desired setting.
This file hopefully becomes unnecessary in a few months.
"""

class qKnowledgeGradient(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""
    Copy of qKnowledgeGradient on
    https://github.com/pytorch/botorch/blob/master/botorch/acquisition/knowledge_gradient.py
    
    with a bug fixed that disabled us using the original qKnowledgeGradient class
    
    =======
    
    Batch Knowledge Gradient using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the inner
    expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models. For a fixed number
    of fantasies, all parts of `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model,
        num_fantasies=64,
        sampler=None,
        objective=None,
        inner_sampler=None,
        X_pending=None,
        current_value=None,
    ):
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None` or a ScalarizedObjective.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
        """
        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(
                num_samples=num_fantasies, resample=False, collapse_batch_dims=True
            )
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super(MCAcquisitionFunction, self).__init__(model=model)
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(
                num_samples=128, resample=False, collapse_batch_dims=True
            )
        if objective is None and model.num_outputs != 1:
            raise UnsupportedError(
                "Must specify an objective when using a multi-output model."
            )
        self.sampler = sampler
        self.objective = objective
        self.set_X_pending(X_pending)
        self.inner_sampler = inner_sampler
        self.num_fantasies = num_fantasies
        self.current_value = current_value

    @t_batch_mode_transform()
    def forward(self, X):
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """

        X_actual, X_fantasies = _split_fantasy_points(
            X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=True
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model, objective=self.objective, sampler=self.inner_sampler
        )

        # make sure to propagate gradients to the fantasy model train inputs
        with botorch.settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        # return average over the fantasy samples
        return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q):
        r"""Get augmented q batch size for one-shot optimzation.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimzation (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full):
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_fantasies, :]


def initialize_q_batch(X, Y, n, eta=1.0):

    r"""[Copy of original botorch function]
    
    Heuristic for selecting initial conditions for candidate generation.

    This heuristic selects points from `X` (without replacement) with probability
    proportional to `exp(eta * Z)`, where `Z = (Y - mean(Y)) / std(Y)` and `eta`
    is a temperature parameter.

    When using an acquisiton function that is non-negative and possibly zero
    over large areas of the feature space (e.g. qEI), you should use
    `initialize_q_batch_nonneg` instead.

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC sampling.
        Y: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.

    Returns:
        A `n x q x d` tensor of `n` `q`-batch initial conditions.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qUCB = qUpperConfidenceBound(model, beta=0.1)
        >>> Xrnd = torch.rand(500, 3, 6)
        >>> Xinit = initialize_q_batch(Xrnd, qUCB(Xrnd), 10)
    """

    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of "
            f"provided samples ({n_samples})"
        )
    elif n == n_samples:
        return X

    Ystd = Y.std()
    if Ystd == 0:
        warnings.warn(
            "All acqusition values for raw samples points are the same. "
            "Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    max_val, max_idx = torch.max(Y, dim=0)
    Z = (Y - Y.mean()) / Ystd
    etaZ = eta * Z
    weights = torch.exp(etaZ)
    while torch.isinf(weights).any():
        etaZ *= 0.5
        weights = torch.exp(etaZ)

    if torch.sum(weights < 0) > 0:
        print(
            'Warning: `weights` in `initialize_q_batch` are not all non-negative '
            'even though a result of `torch.exp`. This could be due to numerical '
            'instability, hence weights are clipped to be non-negative. The original '
            '`weights` tensor had'
        )
        print('shape:            ', weights.shape)
        print('non-zero indices: ', torch.sum(weights < 0))
        print('non-zero values:  ', weights[weights < 0])
        weights = torch.max(weights, torch.tensor(0.0))

    idcs = torch.multinomial(weights, n)
    # make sure we get the maximum
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs]


def gen_batch_initial_conditions(
    acq_function,
    bounds,
    q,
    num_restarts,
    raw_samples,
    options=None):
    r"""[Copy of original botorch function]
    
    Generate a batch of initial conditions for random-restart optimziation.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions).

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
    """
    options = options or {}
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get("batch_limit")
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds = bounds.cpu()
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    dim = bounds.shape[-1] * q
    if dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor
            if dim <= SobolEngine.MAXDIM:
                X_rnd = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed)
            else:
                with manual_seed(seed):
                    # load on cpu
                    X_rnd_nlzd = torch.rand(n * dim, dtype=bounds.dtype).view(
                        n, q, bounds.shape[-1]
                    )
                X_rnd = bounds[0] + (bounds[1] - bounds[0]) * X_rnd_nlzd
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(
                        X_rnd[start_idx:end_idx].to(device=device)
                    ).cpu()
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions


def gen_one_shot_kg_initial_conditions(
    acq_function,
    bounds,
    q,
    num_restarts,
    raw_samples,
    options=None):
    r"""[Copy of original botorch function]
    
    Generate a batch of smart initializations for qKnowledgeGradient.
    This function generates initial conditions for optimizing one-shot KG using
    the maximizer of the posterior objective. Intutively, the maximizer of the
    fantasized posterior will often be close to a maximizer of the current
    posterior. This function uses that fact to generate the initital conditions
    for the fantasy points. Specifically, a fraction of `1 - frac_random` (see
    options) is generated by sampling from the set of maximizers of the
    posterior objective (obtained via random restart optimization) according to
    a softmax transformation of their respective values. This means that this
    initialization strategy internally solves an acquisition function
    maximization problem. The remaining `frac_random` fantasy points as well as
    all `q` candidate points are chosen according to the standard initialization
    strategy in `gen_batch_initial_conditions`.
    Args:
        acq_function: The qKnowledgeGradient instance to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            task features.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        options: Options for initial condition generation. These contain all
            settings for the standard heuristic initialization from
            `gen_batch_initial_conditions`. In addition, they contain
            `frac_random` (the fraction of fully random fantasy points),
            `num_inner_restarts` and `raw_inner_samples` (the number of random
            restarts and raw samples for solving the posterior objective
            maximization problem, respectively) and `eta` (temperature parameter
            for sampling heuristic from posterior objective maximizers).
    Returns:
        A `num_restarts x q' x d` tensor that can be used as initial conditions
        for `optimize_acqf()`. Here `q' = q + num_fantasies` is the total number
        of points (candidate points plus fantasy points).
    Example:
        >>> qKG = qKnowledgeGradient(model, num_fantasies=64)
        >>> bounds = torch.tensor([[0., 0.], [1., 1.]])
        >>> Xinit = gen_one_shot_kg_initial_conditions(
        >>>     qKG, bounds, q=3, num_restarts=10, raw_samples=512,
        >>>     options={"frac_random": 0.25},
        >>> )
    """
    options = options or {}
    frac_random: float = options.get("frac_random", 0.1)
    if not 0 < frac_random < 1:
        raise ValueError(
            f"frac_random must take on values in (0,1). Value: {frac_random}"
        )
    q_aug = acq_function.get_augmented_q_batch_size(q=q)

    # TODO: Avoid unnecessary computation by not generating all candidates
    ics = gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=q_aug,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
    )

    # compute maximizer of the value function
    value_function = _get_value_function(
        model=acq_function.model,
        objective=acq_function.objective,
        sampler=acq_function.inner_sampler,
    )

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=value_function,
        bounds=bounds,
        q=1,
        num_restarts=options.get("num_inner_restarts", 20),
        raw_samples=options.get("raw_inner_samples", 1024),
        return_best_only=False,
    )

    # sampling from the optimizers
    n_value = int((1 - frac_random) * (q_aug - q))  # number of non-random ICs
    eta = options.get("eta", 2.0)
    weights = torch.exp(eta * transforms.standardize(fantasy_vals))
    idx = torch.multinomial(weights, num_restarts * n_value, replacement=True)

    # set the respective initial conditions to the sampled optimizers
    ics[..., -n_value:, :] = fantasy_cands[idx,
                                           0].view(num_restarts, n_value, -1)
    return ics


def _split_fantasy_points(X, n_f):
    r"""[Copy of original botorch function]
    
    Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x 1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy point
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x num_fantasies x d, needs to be num_fantasies x b x 1 x d
    # for batch mode evaluation with batch shape num_fantasies x b.
    # b x num_fantasies x d --> num_fantasies x b x d
    X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
    # num_fantasies x b x 1 x d
    X_fantasies = X_fantasies.unsqueeze(dim=-2)
    return X_actual, X_fantasies


def _get_value_function(
    model,
    objective=None,
    sampler=None):
    r"""Construct value function (i.e. inner acquisition function)."""
    if isinstance(objective, MCAcquisitionObjective):
        return qSimpleRegret(model=model, sampler=sampler, objective=objective)
    else:
        return PosteriorMean(model=model, objective=objective)


