# All ran on 04/01/25
nohup python3 dynamic_w2_barycentre_error.py 0 'heun' > heunx_star.out 2> heunx_star.err &
nohup python3 dynamic_w2_barycentre_error.py 1 'euler' > eulerx_star.out 2> eulerx_star.err &
nohup python3 dynamic_w2_barycentre_error.py 2 'rk4' > rk4x_star.out 2> rk4x_star.err &
