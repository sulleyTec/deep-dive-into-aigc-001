
num_elements=163840000
grid_dim=1480
sudo ncu -o elementwise_add_half_naive --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_add_half_naive

sudo ncu -o elementwise_add_float_naive --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_add_float_naive

sudo ncu -o elementwise_add_half_optimized --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_add_half_optimized

sudo ncu -o elementwise_add_float_optimized --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_add_float_optimized


sudo ncu -o elementwise_mul_half_naive --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_mul_half_naive

sudo ncu -o elementwise_mul_float_naive --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_mul_float_naive

sudo ncu -o elementwise_mul_half_optimized --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_mul_half_optimized

sudo ncu -o elementwise_mul_float_optimized --set full --target-processes  all --replay-mode application --cache-control none ./elementwise ${num_elements} ${grid_dim} elementwise_mul_float_optimized
