"""
TIGER: A translation of the original R code into Python (cmdstanpy).
Author: Chen Chen (Original R), [Your Name for Python translation]
Date Created (Python version): 2025-01-10

Email: cchen22@arizona.edu

Dependencies:
    pandas (for DataFrame)
    numpy
    cmdstanpy
    arviz (optional, for psis-loo if you want to replicate the R "loo" behavior)
    genenet-like partial correlation function in Python (if you want exactly the same prior.pp logic)
"""

import numpy as np
import pandas as pd
import cmdstanpy
import gc
import warnings
import time

from mircid.paths import WORK_ROOT
# If you need advanced LOO features in Python, consider installing ArviZ (pip install arviz)
# import arviz as az


# -----------------------------------------------------------------------------------
# The STAN model code, preserving the original lines from the R string object TIGER_C
# -----------------------------------------------------------------------------------
TIGER_C_STAN = r'''
data {
  int<lower=0> n_genes;                       // Number of genes
  int<lower=0> n_samples;                     // Number of samples
  int<lower=0> n_TFs;                         // Number of TFs
  int<lower=0> n_zero;                        // length of zero elements in P
  int<lower=0> n_ones;                        // length of non-zero elements in P
  int<lower=0> n_negs;                        // length of repression elements in P
  int<lower=0> n_poss;                        // length of activation elements in P
  int<lower=0> n_blur;                        // length of blurred elements in P
  int<lower=0> n_all;                         // length of all elements in P
  matrix[n_genes,n_samples] X;                // Gene expression matrix X
  vector[n_all] P;                            // Prior connection probability
  array[n_zero] int P_zero;                   // index of zero probablity edges
  array[n_ones] int P_ones;                   // index of non-zero prob edges
  array[n_negs] int P_negs;                   // index of repression prob edges
  array[n_poss] int P_poss;                   // index of activation prob edges
  array[n_blur] int P_blur;                   // index of blurred prob edges
  int sign;                                   // use signed prior network or not
  int baseline;                               // inclue baseline term or not
  int psis_loo;                               // use loo to check model or not
  real sigmaZ;                                // prior sd of Z
  real sigmaB;                                // prior sd of baseline
  real a_alpha;                               // hyparameter for inv_gamma
  real b_alpha;
  real a_sigma;                               // hyparameter for inv_gamma
  real b_sigma;
}

transformed data {
  vector[n_genes*n_samples] X_vec;            // gene expression X
  X_vec = to_vector(X);
}

parameters {
  matrix<lower=0>[n_TFs,n_samples] Z;         // TF activity matrix Z
  vector<lower=0>[n_genes] sigma2;            // variances of noise term
  vector[baseline ? n_genes : 0] b0;          // baseline expression for each gene
  vector<lower=0>[sign ? n_blur : 0] alpha0;  // Noise precision of W_blur
  vector<lower=0>[sign ? n_poss : 0] alpha2;  // Noise precision of W_poss
  vector<lower=0>[sign ? n_negs : 0] alpha3;  // Noise precision of W_negs
  vector[sign ? n_blur : 0] beta0;            // Regulatory network blurred edge weight
  vector<upper=0>[sign ? n_negs : 0] beta3;   // Regulatory network negative edge weight
  vector<lower=0>[sign ? n_poss : 0] beta2;   // Regulatory network positive edge weight
  vector<lower=0>[sign ? 0 : n_ones] alpha1;  // Noise precision of W_ones
  vector[sign ? 0 : n_ones] beta1;            // Regulatory network non-zero edge weight
}

transformed parameters {
  vector[sign ? n_negs : 0] W_negs;
  vector[sign ? n_poss : 0] W_poss;
  vector[sign ? n_blur : 0] W_blur;
  vector[sign ? 0 : n_ones] W_ones;

  if (sign) {
    W_negs = beta3.*sqrt(alpha3);  // Regulatory network negative edge weight
    W_poss = beta2.*sqrt(alpha2);  // Regulatory network positive edge weight
    W_blur = beta0.*sqrt(alpha0);  // Regulatory network blurred edge weight
  } else {
    W_ones = beta1.*sqrt(alpha1);  // Regulatory network non-zero edge weight
  }
}

model {
  // local parameters
  vector[n_all] W_vec;                        // Regulatory vector W_vec
  W_vec[P_zero] = rep_vector(0, n_zero);
  if (sign){
    W_vec[P_negs] = W_negs;
    W_vec[P_poss] = W_poss;
    W_vec[P_blur] = W_blur;
  } else {
    W_vec[P_ones] = W_ones;
  }
  matrix[n_genes, n_TFs] W = to_matrix(W_vec, n_genes, n_TFs); // by column
  matrix[n_genes,n_samples] mu = W * Z;       // mu for gene expression X
  if (baseline){
    matrix[n_genes,n_samples] mu0 = rep_matrix(b0, n_samples);
    mu = mu + mu0;
  }
  vector[n_genes*n_samples] X_mu = to_vector(mu);
  vector[n_genes*n_samples] X_sigma = to_vector(rep_matrix(sqrt(sigma2), n_samples));

  // priors
  sigma2 ~ inv_gamma(a_sigma, b_sigma);

  if (baseline){
    b0 ~ normal(0, sigmaB);
  }

  if (sign) {
    // student-t
    alpha2 ~ inv_gamma(a_alpha,b_alpha);
    beta2 ~ normal(0,1);

    alpha3 ~ inv_gamma(a_alpha,b_alpha);
    beta3 ~ normal(0,1);

    alpha0 ~ inv_gamma(a_alpha,b_alpha);
    beta0 ~ normal(0,1);
  } else {
    alpha1 ~ inv_gamma(a_alpha,b_alpha);
    beta1 ~ normal(0,1);
  }

  to_vector(Z) ~ normal(0, sigmaZ);

  // likelihood
  X_vec ~ normal(X_mu, X_sigma);
}

generated quantities {
  vector[psis_loo ? n_genes*n_samples : 0] log_lik;
  if (psis_loo){
    // redefine X_mu, X_sigma
    vector[n_all] W_vec;
    W_vec[P_zero] = rep_vector(0, n_zero);
    if (sign){
      W_vec[P_negs] = W_negs;
      W_vec[P_poss] = W_poss;
      W_vec[P_blur] = W_blur;
    } else {
      W_vec[P_ones] = W_ones;
    }
    matrix[n_genes, n_TFs] W = to_matrix(W_vec, n_genes, n_TFs);
    matrix[n_genes,n_samples] mu = W * Z;
    if (baseline){
      matrix[n_genes,n_samples] mu0 = rep_matrix(b0, n_samples);
      mu = mu + mu0;
    }
    vector[n_genes*n_samples] X_mu = to_vector(mu);
    vector[n_genes*n_samples] X_sigma = to_vector(rep_matrix(sqrt(sigma2), n_samples));

    for (i in 1:(n_genes*n_samples)){
      log_lik[i] = normal_lpdf(X_vec[i] | X_mu[i], X_sigma[i]);
    }
  }
}
'''


def el2adj(el):
    """
    Convert bipartite edge list to adjacency matrix.

    Parameters
    ----------
    el : pd.DataFrame
        An edge list with three columns: (from, to, weight).

    Returns
    -------
    adj : pd.DataFrame
        An adjacency matrix with rows as TFs and columns as genes.
    """
    # R code:
    # el = as.data.frame(el)
    # all.A = unique(el[,1])
    # all.B = unique(el[,2])
    # adj = array(0, dim=c(length(all.A), length(all.B)))
    # rownames(adj) = all.A
    # colnames(adj) = all.B
    # adj[as.matrix(el[,1:2])] = as.numeric(el[,3])
    # return(adj)

    el = el.copy()
    all_A = el.iloc[:, 0].unique()
    all_B = el.iloc[:, 1].unique()
    adj = pd.DataFrame(
        data=0.0,
        index=all_A,
        columns=all_B
    )
    for i in el.index:
        tf = el.iloc[i, 0]
        gene = el.iloc[i, 1]
        w = float(el.iloc[i, 2])
        adj.loc[tf, gene] = w
    return adj


def adj2el(adj):
    """
    Convert a bipartite adjacency matrix to an edge list.

    Parameters
    ----------
    adj : pd.DataFrame
        An adjacency matrix with rows as TFs and columns as genes.

    Returns
    -------
    el : pd.DataFrame
        An edge list with columns: (from, to, weight).
    """
    # R code:
    # el = matrix(NA, nrow(adj)*ncol(adj), 3)
    # el[,1] = rep(row.names(adj), ncol(adj))
    # el[,2] = rep(colnames(adj), each=nrow(adj))
    # ...
    # colnames(el)=c("from","to","weight")
    # return(el)

    rows = adj.index.values
    cols = adj.columns.values
    el_list = []
    for r in rows:
        for c in cols:
            el_list.append([r, c, adj.loc[r, c]])
    el = pd.DataFrame(el_list, columns=["from", "to", "weight"])
    return el


def el2regulon(el):
    """
    Convert a bipartite edge list to a VIPER-required regulon object.

    Parameters
    ----------
    el : pd.DataFrame
        Edge list with columns: (from, to, weight).

    Returns
    -------
    viper_regulons : dict
        A dict of TF -> { 'tfmode': {...}, 'likelihood': [...] }.
    """
    # R code:
    # regulon_list = split(el, el$from)
    # viper_regulons = lapply(regulon_list, function(regulon) {...})
    # return(viper_regulons)
    el = el.copy()
    viper_regulons = {}
    for tf, group_df in el.groupby("from"):
        # tfmode is a dict: gene -> weight
        tfmode_dict = {}
        for idx in group_df.index:
            gene = group_df.loc[idx, "to"]
            weight = group_df.loc[idx, "weight"]
            tfmode_dict[gene] = weight
        # replicate the structure: list(tfmode = tfmode, likelihood = rep(1, length(tfmode)))
        viper_regulons[tf] = {
            "tfmode": tfmode_dict,
            "likelihood": [1.0]*len(tfmode_dict)
        }
    return viper_regulons


def adj2regulon(adj):
    """
    Convert a bipartite adjacency matrix to a VIPER-required regulon object.

    Parameters
    ----------
    adj : pd.DataFrame
        An adjacency matrix, with rows as TFs and columns as genes.

    Returns
    -------
    regulon : dict
        A dict suitable for VIPER usage.
    """
    # R code:
    # el = adj2el(adj)
    # el = el[el[,3]!=0,]
    # regulon = el2regulon(el)
    # return(regulon)

    el = adj2el(adj)
    el = el[el["weight"] != 0]
    regulon = el2regulon(el)
    return regulon


def prior_pp(prior, expr):
    """
    Filter low confident edge signs in the prior network using GeneNet approach
    (partial correlation) to remove sign conflicts.

    Parameters
    ----------
    prior : pd.DataFrame
        A prior network (adjacency matrix) with shape (TFs x Genes).
    expr : pd.DataFrame
        A normalized log-transformed gene expression matrix (Genes x Samples).

    Returns
    -------
    A_ij : pd.DataFrame
        A filtered prior network (adjacency matrix).
    """
    # R code summary (adapted from prior.pp in the original .R):
    #
    # 1. tf = intersect(rownames(prior), rownames(expr))
    # 2. tg = intersect(colnames(prior), rownames(expr))
    # 3. coexp = GeneNet::ggm.estimate.pcor(t(expr[all.gene,]), method = "static")
    #    coexp is partial correlation matrix
    # 4. P_ij = prior[tf, tg]
    # 5. C_ij = coexp[tf, tg] * abs(P_ij)
    # 6. If sign conflict => set prior entry to 1e-6
    # 7. remove all-zero TFs/genes

    # NOTE: In Python, you would need a partial correlation library or to write your own
    # to replicate "GeneNet::ggm.estimate.pcor". Here, we do a placeholder approach.

    tf = list(set(prior.index).intersection(expr.index))
    tg = list(set(prior.columns).intersection(expr.index))
    tf.sort()
    tg.sort()
    if len(tf) == 0 or len(tg) == 0:
        raise ValueError("No matched gene names in the two inputs...")

    # Recreate "coexp" (placeholder: naive correlation).
    # For exact replication of R's "GeneNet" partial correlation, you must implement it.
    # Here, we do a standard correlation for demonstration. Replace with partial corr if needed.
    all_gene = list(set(tf).union(tg))
    all_gene.sort()
    # The original R code uses: coexp = ggm.estimate.pcor(...)
    # We'll just do a correlation matrix as a placeholder.
    sub_expr = expr.loc[all_gene, :]  # shape: len(all_gene) x n_samples
    coexp = sub_expr.T.corr(method='pearson')  # shape: all_gene x all_gene
    np.fill_diagonal(coexp.values, 0)

    # slice prior and coexp
    P_ij = prior.loc[tf, tg].copy()
    C_ij = coexp.loc[tf, tg].copy() * np.abs(P_ij)

    # sign conflict
    sign_P = np.sign(P_ij)
    sign_C = np.sign(C_ij)

    # find conflicts where sign_P * sign_C < 0
    # set prior to 1e-6 where conflict arises
    conflict_idx = np.where((sign_P.values * sign_C.values) < 0)
    # conflict_idx is (rowArray, colArray)
    for r, c in zip(conflict_idx[0], conflict_idx[1]):
        P_ij.iloc[r, c] = 1e-6

    # remove all-zero
    A_ij = P_ij.loc[(P_ij != 0).any(axis=1), (P_ij != 0).any(axis=0)]
    return A_ij


def TIGER(expr,
          prior,
          method="VB",
          TFexpressed=True,
          signed=True,
          baseline=True,
          psis_loo=False,
          seed=123,
          out_path=None,
          out_size=300,
          a_sigma=1,
          b_sigma=1,
          a_alpha=1,
          b_alpha=1,
          sigmaZ=10,
          sigmaB=1,
          tol=0.005):
    """
    TIGER main function (Python translation of the original R code).

    Parameters
    ----------
    expr : pd.DataFrame
        A normalized log-transformed gene expression matrix. Rows = genes, columns = samples.
    prior : pd.DataFrame
        A prior regulatory network in adjacency matrix format. Rows = TFs, columns = target genes.
    method : str, optional
        Method used for Bayesian inference: "VB" or "MCMC". Defaults to "VB".
    TFexpressed : bool, optional
        TF mRNA needs to be expressed or not. Defaults to True.
    signed : bool, optional
        Prior network is signed or not. Defaults to True.
    baseline : bool, optional
        Include baseline in model or not. Defaults to True.
    psis_loo : bool, optional
        Use PSIS-LOO cross validation or not. Defaults to False.
    seed : int, optional
        Random seed. Defaults to 123.
    out_path : str or None, optional
        If not None, path to save the CmdStanVB or CmdStanMCMC object. Defaults to None.
    out_size : int, optional
        Posterior sampling size. Default = 300.
    a_sigma : float, optional
        Hyperparameter of error term. Default = 1.
    b_sigma : float, optional
        Hyperparameter of error term. Default = 1.
    a_alpha : float, optional
        Hyperparameter of edge weight W. Default = 1.
    b_alpha : float, optional
        Hyperparameter of edge weight W. Default = 1.
    sigmaZ : float, optional
        Standard deviation of TF activity Z prior. Default = 10.
    sigmaB : float, optional
        Standard deviation of baseline term prior. Default = 1.
    tol : float, optional
        Convergence tolerance on ELBO for VB. Default = 0.005.

    Returns
    -------
    tiger_fit : dict
        A dictionary containing:
            W : Estimated regulatory network (genes x TFs).
            Z : Estimated TF activities (TFs x samples).
            TF.name, TG.name, sample.name : The used TFs, target genes, and sample names.
            loocv : If psis_loo=True, the table of psis_loo result for model checking (None if not used).
            elpd_loo : The Bayesian LOO estimate of expected log pointwise predictive density.
    """

    # 0. Check data
    sample_name = list(expr.columns)
    if TFexpressed:
        TF_name = sorted(set(prior.index).intersection(expr.index))
    else:
        TF_name = sorted(prior.index)
    TG_name = sorted(set(expr.index).intersection(prior.columns))
    print('>>> Sets of expr.row & prior.col:')
    print(len(set(expr.index)),len(prior.columns))
    print(set(expr.index), prior.columns)

    print("> TG_name len:",len(TG_name))
    print("> TF_name len:",len(TF_name))

    if len(TG_name) == 0 or len(TF_name) == 0:
        raise ValueError("No matched gene names in the two inputs...")

    # 0. prepare stan input
    if signed:
        # if the TF set intersects the TG set
        if len(set(TG_name).intersection(TF_name)) != 0:
            # do prior.pp
            prior2 = prior_pp(prior.loc[TF_name, TG_name], expr)
            # special logic if dimension changed
            if prior2.shape[0] != len(TF_name):
                TFnotExp = list(set(TF_name) - set(prior2.index))
                TFnotExp.sort()
                # add them back with small edges set to 1e-6
                TFnotExpEdge = prior.loc[TFnotExp, prior2.columns].copy()
                TFnotExpEdge[TFnotExpEdge == 1] = 1e-6
                prior2 = pd.concat([prior2, TFnotExpEdge], axis=0)
                prior2 = prior2.sort_index(axis=0)
                # remove all-zero TFs
                prior2 = prior2.loc[(prior2 != 0).any(axis=1)]
            Pmat = prior2
            TF_name = list(Pmat.index)
            TG_name = list(Pmat.columns)
        else:
            Pmat = prior.loc[TF_name, TG_name]
    else:
        Pmat = prior.loc[TF_name, TG_name]

    X = expr.loc[TG_name, :]  # shape: (n_genes, n_samples)
    n_genes = X.shape[0]
    n_samples = X.shape[1]
    n_TFs = Pmat.shape[0]

    # flatten P
    P = Pmat.values.T.flatten()  # R code uses as.vector(t(P)), which flattens row=TG, col=TF
    # but Pmat in Python is shape (TFs, Genes). We want row=TG, col=TF as R does => transpose
    # Then flatten row major => same effect as t(P) in R.

    # find various sets
    P_zero = np.where(P == 0)[0] + 1  # +1 because Stan arrays are 1-based
    P_ones = np.where(P != 0)[0] + 1
    P_negs = np.where(P == -1)[0] + 1
    P_poss = np.where(P == 1)[0] + 1
    # R code uses 1e-6 for "blur"
    P_blur = np.where(P == 1e-6)[0] + 1

    n_zero = len(P_zero)
    n_ones = len(P_ones)
    n_negs = len(P_negs)
    n_poss = len(P_poss)
    n_blur = len(P_blur)
    n_all = len(P)

    sign_int = 1 if signed else 0
    baseline_int = 1 if baseline else 0
    psis_loo_int = 1 if psis_loo else 0

    data_to_model = {
        "n_genes": n_genes,
        "n_samples": n_samples,
        "n_TFs": n_TFs,
        "X": X.values,  # 2D array
        "P": P,         # 1D array
        "P_zero": P_zero.astype(int),
        "P_ones": P_ones.astype(int),
        "P_negs": P_negs.astype(int),
        "P_poss": P_poss.astype(int),
        "P_blur": P_blur.astype(int),
        "n_zero": n_zero,
        "n_ones": n_ones,
        "n_negs": n_negs,
        "n_poss": n_poss,
        "n_blur": n_blur,
        "n_all": n_all,
        "sign": sign_int,
        "baseline": baseline_int,
        "psis_loo": psis_loo_int,
        "sigmaZ": sigmaZ,
        "sigmaB": sigmaB,
        "a_sigma": a_sigma,
        "b_sigma": b_sigma,
        "a_alpha": a_alpha,
        "b_alpha": b_alpha
    }

    # 1. compile the recovered Stan model in the writable work directory.
    stan_dir = WORK_ROOT / "tfa" / "cmdstan"
    stan_dir.mkdir(parents=True, exist_ok=True)
    stan_file = stan_dir / "TIGER_C.stan"
    if not stan_file.exists() or stan_file.read_text(encoding="utf-8") != TIGER_C_STAN:
        stan_file.write_text(TIGER_C_STAN, encoding="utf-8")
    mod = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    # 2. run VB or MCMC
    fit = None
    t_start = time.time()
    if method == "VB":
        # cmdstanpy variational arguments:
        # e.g. mod.variational(data=..., algo='meanfield', seed=..., tol_rel_obj=..., iter=...)
        fit = mod.variational(
            data=data_to_model,
            algorithm="meanfield",
            seed=seed,
            iter=50000,
            tol_rel_obj=tol,
            output_samples=out_size
        )
        print('>>>>>>> COMPLETED VB in ', time.time() - t_start, ' seconds.')
        # print('Model converged: ', fit.diagnose().felbo < tol)
        # print(fit.variational_params_dict)
    elif method == "MCMC":
        # mod.sample(data=..., chains=..., seed=..., ...)
        fit = mod.sample(
            data=data_to_model,
            chains=1,
            seed=seed,
            max_treedepth=10,
            iter_warmup=1000,
            iter_sampling=out_size,
            adapt_delta=0.99
        )
        print('>>>>>>> COMPLETED MCMC in ', time.time() - t_start, ' seconds.')

    else:
        raise ValueError("method must be either 'VB' or 'MCMC'.")

    # 3. optionally save
    if out_path is not None:
        # For cmdstanpy, you can copy or rename the CSV files or
        # use fit.save_csvfiles(...) or simply move them.
        # We'll store the fit object in a pickle for demonstration:
        fit.save_csvfiles(dir=out_path)
        # (You can also pickle the fit object, but be mindful of large files.)

    # 4. posterior distributions
    print("Draw sample from W matrix...")

    # We reconstruct W from "W_negs", "W_poss", "W_blur", or "W_ones"
    # to replicate the R code exactly, we use the summary means.
    # In cmdstanpy: for variational => fit.variational_params_np(name)
    #               for MCMC => use fit.draws_pd(), etc.
    # A simpler approach: both MCMC and VB have a unified "stan_variable" method from 1.0.0 onward.
    # We then compute the average of each variable across draws.

    def get_mean_of_param(param_name):
        """
        Returns the mean across all draws (samples) of the parameter param_name.
        If param_name doesn't exist (like W_negs in unsigned mode if sign=FALSE), returns None.
        """
        try:
            vals = fit.stan_variable(var=param_name, mean=False)
            param_means = vals.mean(axis=0)
            if np.isscalar(param_means):
                param_means = np.array([param_means])  # convert scalar to [scalar]
            # shape could be (samples, dimension)
            # we want the mean across sample axis => axis=0
            return np.mean(vals, axis=0)
        except ValueError:
            return None

    W_pos = np.zeros(n_all)

    if signed:
        W_negs = get_mean_of_param("W_negs")
        if W_negs is not None and n_negs != 0:
            for i, idx in enumerate(P_negs - 1):  # shift back to 0-based
                W_pos[idx] = W_negs[i]

        W_poss = get_mean_of_param("W_poss")
        if W_poss is not None and n_poss != 0:
            for i, idx in enumerate(P_poss - 1):
                W_pos[idx] = W_poss[i]

        W_blur = get_mean_of_param("W_blur")
        if W_blur is not None and n_blur != 0:
            for i, idx in enumerate(P_blur - 1):
                W_pos[idx] = W_blur[i]
    else:
        W_ones = get_mean_of_param("W_ones")
        if W_ones is not None and n_ones != 0:
            for i, idx in enumerate(P_ones - 1):
                W_pos[idx] = W_ones[i]

    W_pos = W_pos.reshape((Pmat.shape[1], Pmat.shape[0]))  # shape: (n_genes, n_TFs)
    # Pmat was shape (TF, Gene), we flattened by row=Gene, col=TF => (Gene x TF).
    # so we reshape: (n_genes, n_TFs)

    gc.collect()

    # 5. posterior for Z
    print("Draw sample from Z matrix...")
    Z_vals = get_mean_of_param("Z")
    # Z shape is (n_TFs, n_samples)
    if Z_vals is None:
        raise ValueError("Z was not found in the stan output!")
    Z_pos = Z_vals  # (n_TFs, n_samples)
    gc.collect()

    # 6. rescale as in R code
    #   IZ = Z_pos*(apply(abs(W_pos),2,sum)/apply(W_pos!=0,2,sum))
    #   IW = t(t(W_pos)*apply(Z_pos,1,sum)/n_samples)
    # Note that W_pos shape is (genes, TFs), Z_pos shape is (TFs, samples).
    # So W_pos != 0 => True/False => sum across rows => we do sum axis=0 for columns.

    absW = np.abs(W_pos)
    sum_absW_cols = absW.sum(axis=0)  # shape (n_TFs,)
    count_nonzero_cols = (W_pos != 0).sum(axis=0)  # shape (n_TFs,)

    # To avoid dividing by zero if a TF has no nonzero edges:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratio1 = sum_absW_cols / count_nonzero_cols  # shape (n_TFs,)

    # IZ = Z_pos * ratio1 (broadcast along rows)
    IZ = Z_pos * ratio1.reshape((n_TFs, 1))

    # For IW, we do sum of Z per TF => apply(Z_pos,1,sum)
    sum_Z_rows = Z_pos.sum(axis=1)  # shape (n_TFs,)
    # IW = t(t(W_pos) * sum_Z_rows / n_samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratio2 = sum_Z_rows / n_samples

    IW = (W_pos.T * ratio2[:, np.newaxis]).T  # shape (genes, TFs)

    # 7. rename
    IW_df = pd.DataFrame(IW, index=TG_name, columns=TF_name)
    IZ_df = pd.DataFrame(IZ, index=TF_name, columns=sample_name)

    # 8. PSIS-LOO if required
    loocv = None
    elpd_loo = None
    if psis_loo:
        print("Pareto Smooth Importance Sampling...")

        # We would replicate R's:
        # loocv = loo::loo(fit$draws("log_lik"), moment_match=TRUE)
        # elpd_loo = loocv$pointwise[,"elpd_loo"]
        #
        # In Python, we can do approximate with ArviZ:
        # draws_log_lik = fit.stan_variable("log_lik")  # shape (draws, n_genes*n_samples)
        # data_for_az = az.from_cmdstanpy(posterior=fit, posterior_predictive="log_lik")
        # loocv = az.loo(data_for_az, pointwise=True)
        # elpd_loo = loocv.loo_pointwise
        #
        # For brevity, store them as None or do a dummy shape:
        # Because the user might not have ArviZ or a direct LOO approach. 
        # We'll store them as None or placeholders.

        loocv = "PSIS-LOO not fully implemented in Python code, please use ArviZ if needed."
        # If you do implement it with ArviZ, you can put the results in "loocv" 
        # and the pointwise ELPD in "elpd_loo".

    # 9. final output
    tiger_fit = {
        "W": IW_df,
        "Z": IZ_df,
        "TF.name": TF_name,
        "TG.name": TG_name,
        "sample.name": sample_name,
        "loocv": loocv,
        "elpd_loo": elpd_loo
    }

    return tiger_fit
