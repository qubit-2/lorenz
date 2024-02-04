       # Lorenz system  ~ yogeshwaran
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the constants and initial conditions
sigma = 10
rho = 28                          # change the values for different result
beta = 8/3
x0, y0, z0 = 1, 1, 1              # initial condt
t0, tf, h = 0, 60, 0.01           # h-(stepsize)

# Define the Lorenz system
def lorenz(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y        #lorenz eqns
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Define the function to solve the Lorenz system using the fourth-order Runge-Kutta method
def solve_lorenz(x0, y0, z0, sigma, rho, beta, t0, tf, h):
    t = np.arange(t0, tf, h)
    n = len(t)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    x[0], y[0], z[0] = x0, y0, z0
    for i in range(n-1):                    # fourth-order Runge-Kutta method
        k1 = h * np.array(lorenz(t[i], [x[i], y[i], z[i]], sigma, rho, beta))
        k2 = h * np.array(lorenz(t[i] + 0.5*h, [x[i] + 0.5*k1[0], y[i] + 0.5*k1[1], z[i] + 0.5*k1[2]], sigma, rho, beta))
        k3 = h * np.array(lorenz(t[i] + 0.5*h, [x[i] + 0.5*k2[0], y[i] + 0.5*k2[1], z[i] + 0.5*k2[2]], sigma, rho, beta))
        k4 = h * np.array(lorenz(t[i] + h, [x[i] + k3[0], y[i] + k3[1], z[i] + k3[2]], sigma, rho, beta))
        x[i+1] = x[i] + (1/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        y[i+1] = y[i] + (1/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        z[i+1] = z[i] + (1/6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return t, x, y, z

# Solving the Lorenz system using the fourth-order Runge-Kutta method
t, x, y, z = solve_lorenz(x0, y0, z0, sigma, rho, beta, t0, tf, h)


# animation function
def animate(i):
    if 2*i>=len(t):             #avoids bound error of x,y,z array
        exit()
    ax.clear()
    ax.set_xlim((np.min(x), np.max(x)))
    ax.set_ylim((np.min(y), np.max(y)))
    ax.set_zlim((np.min(z), np.max(z)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorentz System')
    ax.plot(x, y, z, color='orange',linewidth=0.6)                           #plot of trajectory
    ax.plot(x[2*i], y[2*i], z[2*i], 'o',color='blue',markersize=4)           #plot of dot
    # skipping some points to make animation faster
    ax.text(20, 10, 70, "Time: {}".format(t[2*i]), color='black', backgroundcolor='white')

# Create the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Creating the animation
animation = FuncAnimation(fig, animate, frames=len(t), interval=1)

# Showing the animation
plt.show()
