# Contains the functions for computing the resulting distribution when an assignment instruction is encountered (in state nodes).

# SOGA (defined in SOGA.py)
# |- update_rule
#    |- sym_expr
#    |- update_gaussian

from libSOGAshared import *
from ASGMTListener import *
from ASGMTParser import *
from ASGMTLexer import *
import re

# Cache of parsed assignment functions keyed by (data_id, params_id, var_list, expr_string).
# Safe when expressions don't use loop-variable data array accesses that vary per iteration.
_asgmt_cache = {}

# Matches a non-integer (identifier) inside brackets on the LHS of an assignment.
# e.g.  ang[i] := ...   → matches [i]   → must NOT cache (i is a loop variable)
# e.g.  T[1]  := ...   → no match      → safe to cache (1 is a literal)
# We only inspect the LHS (before ':=') to avoid false positives from gm([...]) in the RHS.
_LHS_VAR_INDEX = re.compile(r'\[([^0-9\]]+)\]')


class AsgmtRule(ASGMTListener):
    
    def __init__(self, var_list, data, params_dict):
        #variables, data and parameters
        self.var_list = var_list
        self.data = data
        self.params = params_dict
        # parameters of the assignment
        self.target = None         # stores the index of the target variable
        self.is_prod = None        # checks whether a term is a product of two vars
        #additional random variables (cannot use a tensors here because different a.r.v.s can have different numbers of components)
        self.aux_pis = []             # stores the weights of auxiliary variables
        self.aux_means = []           # stores the means of auxiliary variables
        self.aux_covs = []            # stores the cov matrices of auxiliary variables
        #function to be applied
        self.func = None           # stores the function
        # Param-dependent coefficients stored separately to avoid stale grad_fn after backward().
        # {var_idx: (numeric_float, param_leaf_tensor)} — recomputed fresh each add_func call.
        self._param_add_coeff = {}
        self._param_add_const = []   # [(numeric_float, param_leaf_tensor)]
        # Per-device cache for add_coeff and add_const to avoid repeated CPU->GPU copies.
        self._add_coeff_on_device = {}
        self._add_const_on_device = {}

    def unpack_rvs(self, term):
        self.aux_pis.append(term.gm().list_()[0].unpack(self.params))
        self.aux_means.append(term.gm().list_()[1].unpack(self.params))
        self.aux_covs.append(torch.pow(term.gm().list_()[2].unpack(self.params),2))
        
    def enterAssignment(self, ctx):
        self.target = self.var_list.index(ctx.symvars().getVar(self.data))
   
       
    def enterAdd(self, ctx):
        # a product is a single add_term in which the terms are both variables
        if len(ctx.add_term())==1 and len(ctx.add_term(0).term()) == 2:
            self.is_prod = 1
            for term in ctx.add_term(0).term():
                self.is_prod = self.is_prod*term.is_var(self.data)
        if self.is_prod:
            self.mul_idx = []
        else:
            self.add_coeff = torch.zeros(len(self.var_list))
            self.add_const = torch.tensor(0.)
    
    def enterAdd_term(self,ctx):
        # product between variables
        if self.is_prod:
            for term in ctx.term():
                if not term.gm() is None:
                    self.unpack_rvs(term)
                    self.mul_idx.append(int(len(self.var_list)+len(self.aux_pis)-1))
                elif not term.symvars() is None:
                    self.mul_idx.append(self.var_list.index(term.symvars().getVar(self.data)))
            self.func = partial(mul_func,self)
        # linear combination
        else:
            # Track numeric part (Python float) and param leaf tensor separately.
            # This avoids storing non-leaf tensors (e.g. 1.0 * param) whose backward
            # graph gets freed after the first backward(), causing errors on iteration 2+.
            numeric_coeff = 1.0
            param_leaf = None
            var_idx = None
            for term in ctx.term():
                if term.sub() is not None:
                    numeric_coeff = -numeric_coeff
                if term.is_const(self.data):
                    val = term.getValue(self.data, self.params)
                    if isinstance(val, torch.Tensor) and val.requires_grad:
                        # Leaf param tensor — store separately, don't apply numeric ops now
                        param_leaf = val if param_leaf is None else param_leaf * val
                    else:
                        v = val.item() if isinstance(val, torch.Tensor) else float(val)
                        numeric_coeff = numeric_coeff * v
                elif not term.symvars() is None:
                    var_idx = self.var_list.index(term.symvars().getVar(self.data))
                elif not term.gm() is None:
                    self.unpack_rvs(term)
                    var_idx = len(self.add_coeff) + 1
            if not var_idx is None:
                if param_leaf is not None:
                    # Store leaf + scalar; add_func recomputes numeric_coeff * param_leaf each call
                    self._param_add_coeff[var_idx] = (numeric_coeff, param_leaf)
                    placeholder = torch.tensor(0.)
                    if var_idx < len(self.add_coeff):
                        self.add_coeff[var_idx] = placeholder
                    else:
                        self.add_coeff = torch.hstack([self.add_coeff, placeholder])
                else:
                    coeff_t = torch.tensor(float(numeric_coeff))
                    if var_idx < len(self.add_coeff):
                        self.add_coeff[var_idx] = coeff_t
                    else:
                        self.add_coeff = torch.hstack([self.add_coeff, coeff_t])
            else:
                if param_leaf is not None:
                    self._param_add_const.append((numeric_coeff, param_leaf))
                else:
                    self.add_const = self.add_const + float(numeric_coeff)
                                
    def exitAdd(self, ctx):
        if not self.is_prod:
            if not torch.all(self.add_coeff == 0) or self._param_add_coeff:
                self.func = partial(add_func, self)
            # this part makes the distribution non differentiable but is needed for the smoother
            else:
                self.func = partial(const_func, self)    # check if this is needed after the new smoother is ready
                
    
def asgmt_parse(var_list, expr, data, params_dict):
    """ Parses expr using ANTLR4. Returns a function """
    lexer = ASGMTLexer(InputStream(expr))
    stream = CommonTokenStream(lexer)
    parser = ASGMTParser(stream)
    tree = parser.assignment()
    asgmt_rule = AsgmtRule(var_list, data, params_dict)
    walker = ParseTreeWalker()
    walker.walk(asgmt_rule, tree) 
    return asgmt_rule.func
        
        
def update_rule(dist, expr, data, params_dict):
    """ Applies expr to dist. It first parses expr using the function asgmt_parse, implemented as an ANTLR listener. asgmt_parse returns a function rule_func, such that, rule_func(GaussianMix) returns a new GaussianMix object obtained applying expr to the initial distribution. rule_func is applied to each component of dist, and the resulting Gaussian mixtures are stored in a single GaussianMix object."""
    if expr == 'skip':
        return dist
    # Skip caching only when the LHS target variable has a non-integer array index (e.g. ang[i])
    # whose value is a loop variable looked up from data at parse time.  The expression string
    # is the same every iteration but resolves to a different target variable, so caching gives
    # wrong results.  Fixed-index accesses (e.g. T[1]) are safe to cache — check only the LHS.
    if '[' in expr:
        lhs = expr.split(':=')[0]
        if _LHS_VAR_INDEX.search(lhs):
            return asgmt_parse(dist.var_list, expr, data, params_dict)(dist)
    # Use id(params_dict) as cache key so different optimization runs don't share stale entries.
    params_id = id(params_dict) if params_dict else None
    data_id = id(data) if data else None
    cache_key = (data_id, params_id, tuple(dist.var_list), expr)
    if cache_key not in _asgmt_cache:
        _asgmt_cache[cache_key] = asgmt_parse(dist.var_list, expr, data, params_dict)
    return _asgmt_cache[cache_key](dist)
    
def add_func(self, dist):

    device = dist.get_device() # pytorch tensor get_device returns GPU id 0,1,etc or -1 (eg for CPU)
    if device >= 0:
        device = f'cuda:{device}'
    else:
        device = 'cpu'

    # Build add_coeff: cached device copy + fresh param contributions each call.
    # The base (non-param) coefficients are cached per device to avoid repeated CPU->GPU copies.
    # Param-dependent entries are recomputed fresh to avoid stale grad_fn after backward().
    if device not in self._add_coeff_on_device:
        self._add_coeff_on_device[device] = self.add_coeff.to(device)
    if self._param_add_coeff:
        add_coeff = self._add_coeff_on_device[device].clone()
        for var_idx, (numeric, param_leaf) in self._param_add_coeff.items():
            e_i = torch.zeros_like(add_coeff)
            e_i[var_idx] = numeric
            add_coeff = add_coeff + e_i * param_leaf.to(device)
    else:
        add_coeff = self._add_coeff_on_device[device]

    if self._param_add_const:
        # Has param contributions — build fresh each call
        add_const_val = self.add_const
        if isinstance(add_const_val, (int, float)):
            add_const_val = torch.tensor(add_const_val, device=device, dtype=dist.gm.mu.dtype)
        else:
            add_const_val = add_const_val.to(device)
        for numeric, param_leaf in self._param_add_const:
            add_const_val = add_const_val + numeric * param_leaf.to(device)
    else:
        # Pure constant — cache per device
        if device not in self._add_const_on_device:
            val = self.add_const
            if isinstance(val, (int, float)):
                self._add_const_on_device[device] = torch.tensor(val, device=device, dtype=dist.gm.mu.dtype)
            else:
                self._add_const_on_device[device] = val.to(device)
        add_const_val = self._add_const_on_device[device]

    i = self.target
    old_dim = dist.gm.n_dim()

    # STEP 1: considers all possible combinations of components of the auxiliary variables
    extended_gm = extend_dist(self, dist)   # see libSOGAshared

    # STEP 2: computes vectorially the new means and covariance matrices
    # Optimize: compute matmul once and reuse
    extended_mu = torch.clone(extended_gm.mu)
    extended_mu[:, i] = torch.matmul(extended_gm.mu, add_coeff) + add_const_val
    extended_sigma = torch.clone(extended_gm.sigma)
    # Compute matmul once and reuse for both assignments
    sigma_coeff = torch.matmul(add_coeff, extended_gm.sigma)  # Shape: (n_comp, n_dim)
    extended_sigma[:, i, :] = sigma_coeff
    extended_sigma[:, :, i] = sigma_coeff
    # Compute diagonal element efficiently
    extended_sigma[:, i, i] = torch.matmul(sigma_coeff, add_coeff.reshape(-1,1)).flatten()
    new_dist = DistGPU(dist.var_list, GaussianMixGPU(extended_gm.pi, extended_mu[:, :old_dim], extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()

    return new_dist


def mul_func(self, dist):

    i = self.target
    j, k = self.mul_idx
    old_dim = dist.gm.n_dim()
    
    # STEP 1: considers all possible combinations of components of the auxiliary variables
    extended_gm = extend_dist(self, dist)   # see libSOGAshared

    # STEP 2: computes mean and covariance matrix for the extended component
    extended_mu = torch.clone(extended_gm.mu)
    extended_mu[:,i] = extended_gm.sigma[:,j,k] + extended_gm.mu[:,j]*extended_gm.mu[:,k]
    extended_sigma = torch.clone(extended_gm.sigma)
    extended_sigma[:, i, :] = extended_sigma[:, :, i] = extended_gm.mu[:,j].reshape(-1, 1)*extended_gm.sigma[:,k,:] + extended_gm.mu[:,k].reshape(-1, 1)*extended_gm.sigma[:,j,:]
    extended_sigma[:, i, i] = torch.pow(extended_gm.sigma[:, j, k], 2)  + 2*extended_gm.sigma[:,j,k]*extended_gm.mu[:, j]*extended_gm.mu[:, k] + extended_gm.sigma[:,j,j]*extended_gm.sigma[:,k,k] + extended_gm.sigma[:,j,j]*torch.pow(extended_gm.mu[:,k], 2) + extended_gm.sigma[:,k,k]*extended_gm.mu[:,j]**2
    new_dist = DistGPU(dist.var_list, GaussianMixGPU(extended_gm.pi, extended_mu[:, :old_dim], extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()

    return new_dist
    

def const_func(self, dist):
    device = dist.get_device()
    if device >= 0:
        device = f'cuda:{device}'
    else:
        device = 'cpu'
    
    i = self.target
    new_mu = torch.clone(dist.gm.mu)
    new_mu[:, i] = self.add_const*torch.ones(len(new_mu[:,i]), device=device)
    new_sigma = torch.clone(dist.gm.sigma)
    new_sigma[:, i, :] = new_sigma[:, :, i] = torch.zeros(new_sigma[:,:,i].shape, device=device)
    new_dist = DistGPU(dist.var_list, GaussianMixGPU(dist.gm.pi, new_mu, new_sigma))
    return new_dist
            

    
    
    