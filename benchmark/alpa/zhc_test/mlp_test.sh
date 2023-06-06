ray stop --force
ray start --head
# python3 test_mlp_1.py
# XLA_PYTHON_CLIENT_PREALLOCATE=false python3 test_mlp_1.py
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 test_mlp_2.py
# python3 test_mlp_2.py