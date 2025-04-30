# Contains the functions for computing the resulting distribution when an assignment instruction is encountered (in state nodes).

# SOGA (defined in SOGA.py)
# |- update_rule
#    |- sym_expr
#    |- update_gaussian

from libSOGAshared import *
from ASGMTListener import *
from ASGMTParser import * 
from ASGMTLexer import *


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
            # collects the coefficients of the linear combination
            coeff = torch.tensor(1.)
            var_idx = None
            for term in ctx.term():
                if term.sub() is not None:
                    coeff = -1*coeff
                else:
                    coeff = 1*coeff
                if term.is_const(self.data):
                    coeff = coeff*term.getValue(self.data, self.params)
                elif not term.symvars() is None:
                    var_idx = self.var_list.index(term.symvars().getVar(self.data))
                elif not term.gm() is None:
                    self.unpack_rvs(term)
                    var_idx = len(self.add_coeff) + 1
            if not var_idx is None:
                if var_idx < len(self.add_coeff):
                    self.add_coeff[var_idx] = coeff
                else:
                    self.add_coeff = torch.hstack([self.add_coeff, coeff])
            else:
                self.add_const = self.add_const + coeff
                                
    def exitAdd(self, ctx):
        if not self.is_prod:
            if not torch.all(self.add_coeff == 0):
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
    else:
        rule_func = asgmt_parse(dist.var_list, expr, data, params_dict)  
        return rule_func(dist)
    
def add_func(self, dist):
    
    i = self.target
    old_dim = dist.gm.n_dim()

    # STEP 1: considers all possible combinations of components of the auxiliary variables
    extended_gm = extend_dist(self, dist)   # see libSOGAshared

    # STEP 2: computes vectorially the new means and covariance matrices 
    extended_mu = torch.clone(extended_gm.mu)
    extended_mu[:, i] = torch.matmul(extended_gm.mu, self.add_coeff) + self.add_const
    extended_sigma = torch.clone(extended_gm.sigma)
    extended_sigma[:, i, :] = extended_sigma[:, :, i] = torch.matmul(self.add_coeff, extended_gm.sigma)
    extended_sigma[:, i, i] = torch.matmul(torch.matmul(self.add_coeff, extended_gm.sigma), self.add_coeff.reshape(-1,1)).flatten()  
    new_dist = Dist(dist.var_list, GaussianMix(extended_gm.pi, extended_mu[:, :old_dim], extended_sigma[:, :old_dim, :old_dim]))
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
    new_dist = Dist(dist.var_list, GaussianMix(extended_gm.pi, extended_mu[:, :old_dim], extended_sigma[:, :old_dim, :old_dim]))
    new_dist.gm.delete_zeros()

    return new_dist
    

def const_func(self, dist):
    i = self.target
    new_mu = torch.clone(dist.gm.mu)
    new_mu[:, i] = self.add_const*torch.ones(len(new_mu[:,i]))
    new_sigma = torch.clone(dist.gm.sigma)
    new_sigma[:, i, :] = new_sigma[:, :, i] = torch.zeros(new_sigma[:,:,i].shape)
    new_dist = Dist(dist.var_list, GaussianMix(dist.gm.pi, new_mu, new_sigma))
    return new_dist
            

    
    
    