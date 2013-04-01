float perp = 0.0;
for (int i = 0; i < N; i++) {
  int d = d_v_length(i, 0);
  int v = d_v_length(i, 1);
  int length = d_v_length(i, 0);
  float p = 0.0;
  for (int j = 0; j < T; j++) {
    p += theta_D_T(d, j) * phi_T_W(j, v);
  }
  perp -= log(p);
}
return_val = exp(perp / N);
