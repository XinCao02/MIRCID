
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
