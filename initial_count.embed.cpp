for(int i = 0; i < N; i++) {
  int d = d_v_length(i,0);
  int v = d_v_length(i,1);
  int length = d_v_length(i,2);
  int j = z(i);
  n_D_T(d,j) += length;
  n_W_T(v,j) += length;
  n_D(d) += length;
  n_T(j) += length;
}
