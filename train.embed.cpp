for (int i = 0; i < N; i++) {
  float Q_i[T];
  int z_i = z(i);
  int d = w_N(i, 0);
  int v = w_N(i, 1);
  for(int j = 0; j < T; j++) {
    int n_v_j_i = n_W_T(v,j);
    int n_d_j_i = n_D_T(d,j);
    int n_j_i = n_T(j);
    int n_d_i = n_D(d) - 1;
    if (z_i == j) {
      n_v_j_i--;
      n_d_j_i--;
      n_j_i--;
    }
    double q_i = (n_v_j_i + beta) * (n_d_j_i + alpha) /
      (n_j_i + W * beta) / (n_d_i + T * alpha);
    double Q_i_sum = (j > 0)? Q_i[j-1] : 0;
    Q_i[j] = Q_i_sum + q_i;
  }
  float u = random() / (float)RAND_MAX;
  int j;
  for (j = 0; j < T; j++) {
    if(Q_i[j] / Q_i[T-1] >= u) break;
  }
  if (z_i != j) {
    z(i) = j;
    n_D_T(d,z_i) -= 1;
    n_W_T(v,z_i) -= 1;
    n_T(z_i) -= 1;
    n_D_T(d,j) += 1;
    n_W_T(v,j) += 1;
    n_T(j) += 1;
  }
}
