from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange
import math

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce
import itertools

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y

def get_2x2_perm_matrices(device='cpu'):
    """
    Returns the two 2x2 permutation matrices: identity and swap.
    Shape: (2, 2, 2) - [num_perms, 2, 2]
    """
    # P_0 = Identity: [[1,0],[0,1]]
    # P_1 = Swap:     [[0,1],[1,0]]
    perms = torch.tensor([
        [[1., 0.], [0., 1.]],  # Identity
        [[0., 1.], [1., 0.]]   # Swap
    ], dtype=torch.float32, device=device)
    return perms


def factorize_into_twos(n: int):
    """
    Factorize n into a product of factors, preferring 2s.
    For n = 2^k, returns [2, 2, ..., 2] (k times).
    For other n, returns a list of factors that multiply to n.
    """
    if n == 1:
        return []
    
    factors = []
    remaining = n
    
    # First extract all factors of 2
    while remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    
    # If there's a remainder > 1, we need to handle non-power-of-2
    # Note that we actually do not support a n that is not a power of 2, as the current implementation is not efficient for other cases. This will be supported in a future version.
    if remaining > 1:
        factors.append(remaining)
    
    return factors


def get_all_permutations(n: int):
    """
    Generate all n × n permutation matrices, returned as shape (n!, n, n)
    """
    assert n >= 1, "n must be a positive integer"

    perms = list(itertools.permutations(range(n)))
    index = torch.tensor(perms, dtype=torch.long, device="cpu")

    eye = torch.eye(n, dtype=torch.float32, device="cpu")
    perm_mats = eye[index]  # (n!, n, n)

    return perm_mats


# Cache for permutation matrices
perm_mats_2x2 = {}
perm_mats_general = {}

def get_cached_2x2_perms(device):
    """Get cached 2x2 permutation matrices for given device"""
    dev_key = str(device)
    if dev_key not in perm_mats_2x2:
        perm_mats_2x2[dev_key] = get_2x2_perm_matrices(device)
    return perm_mats_2x2[dev_key]


# main functions

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed = False,
    dim = None,
    disable = False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), '`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added'

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams = True)
    else:
        expand_fn = Reduce(pattern = 'b ... -> (b s) ...', reduction = 'repeat', s = num_streams)

    reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = 'sum', s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = KromHC if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main classes

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(
        self,
        residuals
    ):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,

    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


class KromHC(Module):
    """
    Kronecker Low-Rank Hyper-Connections (KromHC)
    
    This implements the tensorial manifold-constrained hyper-connections where
    the Hres matrix is represented as a Kronecker product of small doubly stochastic
    factor matrices, following the Tucker decomposition structure.
    
    For n residual streams where n = prod(i_k):
    - Instead of n! permutation combinations (factorial complexity)
    - Uses K factor matrices U_1, ..., U_K, each i_k x i_k
    - Each U_k is a convex combination of i_k! permutation matrices
    - Hres = U_1 ⊗ U_2 ⊗ ... ⊗ U_K (Kronecker product)
    
    For n = 2^K (power of 2), each factor is 2x2 with only 2 permutations
    """
    
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index = None,
        channel_first = False,
        dropout = 0.,
        residual_transform: Module | None = None,
        add_branch_out_to_residual = True,
        num_input_views = 1,
        depth_residual_fn = add,
        num_fracs = 1,
    ):
        super().__init__()

        #### We assume num_fracs = 1, num_input_views = 1 ###

        self.branch = branch
        assert num_fracs >= 1
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f = num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')
        assert divisible_by(dim, num_fracs), f'feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})'
        
        dim //= num_fracs

        assert num_residual_streams >= 2, '`num_residual_streams` must be at least 2'
        assert num_residual_streams & (num_residual_streams - 1) == 0, f'`num_residual_streams` must be a power of 2, got {num_residual_streams}'

        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        self.num_fracs = num_fracs

        # width num residual streams
        self.norm = RMSNorm(dim * num_residual_streams_fracs)

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # Factorize num_residual_streams into 2s
        self.factors = factorize_into_twos(num_residual_streams)
        self.num_factors = len(self.factors)
        
        # For each factor i_k, we need i_k! coefficients
        # For factor=2, that's 2 coefficients (identity vs swap)
        self.factor_perms = []
        total_res_coeffs = 0
        for f in self.factors:
            num_perms = math.factorial(f)
            self.factor_perms.append(num_perms)
            total_res_coeffs += num_perms
        
        # Cache permutation matrices for non-2 factors (this will be needed for supporting n=non-power of 2s in the future)
        for f in self.factors:
            if f > 2 and (f, "cpu") not in perm_mats_general:
                perm_mats_general[(f, "cpu")] = get_all_permutations(f).to("cpu")
        
        init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
        init_alpha0[init_residual_index, :] = 1.

        # H_res parameters for Kronecker structure
        # Initialize to imitate identity (i.e., use the first permutation in each 2x2 factor)
        init_alpha1 = torch.ones(total_res_coeffs * num_fracs) * -8
        # Set first permutation coefficient of each factor to 0 (imitates identity P_0)
        coeff_idx = 0
        for num_perms in self.factor_perms:
            init_alpha1[coeff_idx] = 0.  # Imitates first perm of this factor → identity
            coeff_idx += num_perms

        # Combined static alpha
        self.static_alpha = nn.Parameter(cat([
            init_alpha0.view(-1),
            init_alpha1
        ], dim=-1))

        # Dynamic alpha function
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim * num_residual_streams, 
                num_fracs * (total_res_coeffs + num_residual_streams * num_input_views)
            )
        )

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)

        # Shared learnable scalar alpha_l^{res} for all factors k
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)
        
        self.total_res_coeffs = total_res_coeffs

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            beta_init = torch.ones(num_residual_streams_fracs) * -1.
            beta_init[init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)

            dynamic_beta_shape = (
                dim * num_residual_streams,
                num_fracs * num_residual_streams
            )
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))

            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)

        self.channel_first = channel_first

        self.residual_transform = default(residual_transform, nn.Identity())

        self.depth_residual_fn = depth_residual_fn

    def _get_factor_perms(self, factor_size, device):
        """Get permutation matrices for a given factor size"""
        if factor_size == 2:
            return get_cached_2x2_perms(device)
        else:
            dev_key = str(device)
            if (factor_size, dev_key) not in perm_mats_general:
                perm_mats_general[(factor_size, dev_key)] = get_all_permutations(factor_size).to(device)
            return perm_mats_general[(factor_size, dev_key)]

    def _build_kronecker_hres(self, dynamic_coeffs, static_coeffs, device):
        """
        Build the Hres matrix using Kronecker product of factor matrices.
        
        Uses a shared α_l^{res} scalar to scale all dynamic coefficients uniformly:
            a_l^k = Softmax(α_l^{res} * x_l' @ W_l^{res,k} + b_l^{res,k})
        
        
        Args:
            dynamic_coeffs: Tensor of shape (..., total_res_coeffs) containing dynamic (x'W) coefficients
            static_coeffs: Tensor of shape (total_res_coeffs,) containing static bias coefficients
            device: Device for permutation matrices
            
        Returns:
            Tensor of shape (..., n, n) where n = num_residual_streams
        """
        if len(self.factors) == 0:
            return dynamic_coeffs.new_ones(dynamic_coeffs.shape[:-1] + (1, 1))
        
        # Apply shared residual scale a_l^{res} to all factors
        combined_coeffs = self.residual_scale * dynamic_coeffs + static_coeffs
        
        # Check if all factors are 2x2
        all_2x2 = all(f == 2 for f in self.factors)
        
        if all_2x2:
            batch_shape = combined_coeffs.shape[:-1]
            coeffs_reshaped = combined_coeffs.view(*batch_shape, self.num_factors, 2)

            weights = F.softmax(coeffs_reshaped, dim=-1) 
            p = weights[..., 0]
            
            # For 2x2: convex combo of I and swap gives [[p, 1-p], [1-p, p]]
            one_minus_p = 1.0 - p  # (..., num_factors)
            
            row0 = torch.stack([p, one_minus_p], dim=-1)
            row1 = torch.stack([one_minus_p, p], dim=-1)
            all_factor_matrices = torch.stack([row0, row1], dim=-2)
            
            # Do Kronecker product iteratively
            result = all_factor_matrices[..., 0, :, :]
            for k in range(1, self.num_factors):
                mat = all_factor_matrices[..., k, :, :]  # (..., 2, 2)
                result_exp = result.unsqueeze(-1).unsqueeze(-3)  # (..., a1, 1, a2, 1)
                mat_exp = mat.unsqueeze(-4).unsqueeze(-2)        # (..., 1, b1, 1, b2)
                kron = result_exp * mat_exp
                result = kron.reshape(*batch_shape, result.shape[-2] * 2, result.shape[-1] * 2)
            
            return result
        else:
            # Fallback for non-2x2 factors (this will be tested and supported for n=non-power of 2s in the future)
            factor_matrices = []
            coeff_idx = 0
            
            for k, (factor_size, num_perms) in enumerate(zip(self.factors, self.factor_perms)):
                factor_coeffs = combined_coeffs[..., coeff_idx:coeff_idx + num_perms]
                coeff_idx += num_perms
                
                perms = self._get_factor_perms(factor_size, device)
                weights = F.softmax(factor_coeffs, dim=-1)
                U_k = einsum(weights, perms, '... r, r i j -> ... i j')
                factor_matrices.append(U_k)
            
            result = factor_matrices[0]
            for mat in factor_matrices[1:]:
                result_exp = rearrange(result, '... a1 a2 -> ... a1 1 a2 1')
                mat_exp = rearrange(mat, '... b1 b2 -> ... 1 b1 1 b2')
                kron = result_exp * mat_exp
                result = rearrange(kron, '... a b c d -> ... (a b) (c d)')
            
            return result

    def width_connection(
        self,
        residuals
    ):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # handle channel first
        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        # split out fractions
        residuals = self.split_fracs(residuals)

        # split out streams
        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = streams)

        # norm
        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s = streams)
        normed = self.norm(normed)

        if self.add_branch_out_to_residual:
            fused_weights = cat([self.dynamic_alpha_fn, self.dynamic_beta_fn], dim=-1)
        else:
            fused_weights = self.dynamic_alpha_fn
        combined_weight = normed @ fused_weights
        
        # Split the fused output into alpha and beta parts
        alpha_size = self.dynamic_alpha_fn.shape[-1]
        wc_weight = combined_weight[..., :alpha_size]
        
        psize = self.num_input_views * streams
        dynamic_pre, dynamic_residual = wc_weight[..., :psize], wc_weight[..., psize:]
        static_pre, static_residual = self.static_alpha[:psize], self.static_alpha[psize:]

        device = combined_weight.device
        
        alpha_residual = self._build_kronecker_hres(dynamic_residual, static_residual, device)
        alpha_residual = self.split_fracs(alpha_residual)

        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre
        alpha_pre = rearrange(alpha_pre, '... (f s v) -> ... s f v', v=self.num_input_views, f=self.num_fracs)
        alpha_pre = alpha_pre.sigmoid()

        alpha = cat((alpha_pre, alpha_residual), dim=-1)  # (..., f, s, f, v+s)

        # beta for weights from branch output back to residual streams
        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = combined_weight[..., alpha_size:]
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s=streams)

            dynamic_beta = dc_weight * self.h_post_scale

            static_beta = rearrange(self.static_beta, '... (s f) -> ... s f', s=streams)

            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2  # sigmoid * 2 for "H^post"

        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :], mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        branch_input = self.merge_fracs(branch_input)
        
        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta=beta)

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta
    ):
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')

        output = rearrange(output, 'b ... s d -> (b s) ... d')

        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):

            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

KromHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
KromHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False,
        expand_to_streams = False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):

        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> (b s) ...', s = self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s = self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s = self.num_streams)

        return residuals


class AttentionPoolReduceStream(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False
    ):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim = -2)

        residuals = reduce(residuals * attn, 'b ... s d -> b ... d', 'sum')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return residuals
