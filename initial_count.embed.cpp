for(int i = 0; i < N; i++) {
  int d = w_N(i,0);
  int v = w_N(i,1);
  int j = z(i);
  n_D_T(d,j) += 1;
  n_W_T(v,j) += 1;
  n_D(d) += 1;
  n_T(j) += 1;
}
