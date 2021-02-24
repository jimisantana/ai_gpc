import torch
import timeit
import matplotlib.pyplot as plt

def fn():
    runtimes = []
    threads = [1] + [t for t in range(2, 17, 2)]

    for t in threads:
        torch.set_num_threads(t)
        r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
        runtimes.append(r)

    print(threads, runtimes)

    plt.figure()
    plt.plot(threads, runtimes)
    plt.show()

fn()
