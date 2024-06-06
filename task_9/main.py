import numpy
from matplotlib import pyplot as plt, cm
from heat_equation import HeatEquation


def main():
    he = HeatEquation(
        k=1 / (numpy.pi ** 2),
        f=lambda x, t: 0,
        a=1,
        t=1,
        mu=lambda x: 1 / (numpy.pi ** 2) * numpy.sin(numpy.pi * x),
        mu1=lambda x: 0,
        mu2=lambda x: 0,
    )

    x_exp, t_exp, u_exp = he.solve_with_explicit_scheme(17, 30)
    x_imp, t_imp, u_imp = he.solve_with_implicit_scheme(30, 30)

    x1, t1 = numpy.meshgrid(x_exp, t_exp)
    x2, t2 = numpy.meshgrid(x_imp, t_imp)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    axes1 = fig.add_subplot(1, 2, 1, projection='3d')
    axes1.plot_surface(x1, t1, u_exp, cmap=cm.plasma)
    axes1.set_title("Явная схема")
    axes2 = fig.add_subplot(1, 2, 2, projection='3d')
    axes2.plot_surface(x2, t2, u_imp, cmap=cm.viridis)
    axes2.set_title("Неявная схема")

    plt.show()

    # u1_i = he.solve_with_implicit_scheme(10, 30)
    # fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    # x1, t1 = np.meshgrid(u1_i[0], u1_i[1])
    # surf1 = ax1.plot_surface(x1, t1, u1_i[2], cmap=cm.viridis, alpha=0.7)
    # ax1.view_init(elev=30, azim=45, roll=0)
    # plt.show()
    #
    # u1_i = he.solve_with_explicit_scheme(20, 30)
    # fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    # x1, t1 = np.meshgrid(u1_i[0], u1_i[1])
    # surf1 = ax1.plot_surface(x1, t1, u1_i[2], cmap=cm.viridis, alpha=0.7)
    # ax1.view_init(elev=30, azim=45, roll=0)
    # plt.show()

if __name__ == "__main__":
    main()
