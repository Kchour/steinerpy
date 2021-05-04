def test_args_kwargs(*args, **kwargs):
    for k in kwargs.keys():
        print(k)


test_args_kwargs(fire="hot", ice="cold")