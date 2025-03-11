## Library Imports ##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import animation

## Initial Configurations - User Defined ## 

# Run the N initial conditions simultaneously, where N = my_range. No-
# te that the simulation time increases exponentially with N. # Higher
# numbers are recommended to fully describe the chaotic behavior,a s 
# very small numbers of N can lead to incorret conclusions (less flips
# may be detected) 
# N = 50   → 2 minutes
# N = 100  → 5 minutes
# N = 500  → 2 hours
# N = 1000 → 6 hours
# N = 1500 → 16 hours
N = 50

# Define the dpi of the plots. Note that this is configured to be proportional to N.
# It is recommended to use 100 dpi with N = 50,100 and 250 dpi with N=1000,1500 
dpi_figures = 100

## Numerical Methods - Implementation ##

def RK4_method(f, y0, t, args=()):
    """Implementation of Runge-Kutta (4th order) numerical method"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

def Euler_method(f, y0, t, args=()):
    """Implementation of Euler numerical method"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        y[i+1] = y[i] + f(y[i], t[i], *args)*h
    return y

## Numerical Methods - Tests ##

def test(f_method1, f_method2, f_test, y0, t, f_test_latex, f_analytical):    
    # Function to compare two numerical methods (inputs 1 and 2) with an
    # ODE (input 3, 4) for a time (t). Also expects the function as a s-
    # tring (to display a title, input 5) and the analytical solution 
    # (input 6).

    y_RK    = f_method1(f_test, y0, t, args=())
    y_euler = f_method2(f_test, y0, t, args=())
    y_analytical = f_analytical

    plt.subplots(figsize=(10,8), dpi=300)
    plt.plot(t,y_euler,'r.',t,y_RK,'b+',t,y_analytical,'k')
    plt.grid(True)
    plt.legend(['Euler\'s Method, dt=' + str(max(t)/len(t)), 'RK4 Method, dt=' + str(max(t)/len(t)) ,'Analytical Solution'])
    plt.title("Solution of " + f_test_latex)
    plt.ylabel('y')
    plt.xlabel('time, t (seconds)')
    plt.savefig("Numerical Methods - Test" + str(y0)+".png",bbox_inches='tight', dpi=300)
    plt.show()

# Test 1
##f = lambda y,t: y**2
#t = np.linspace(0,20,160)
#y0 = [-1]
#f_analytical = - 1 / (t + 1)
#f_test_latex="$y'=y^2 , y(0)=-11$ \n (Analytical: $ y=\\frac{-1}{t + 1} $)"
#test(RK4_method, Euler_method, f, y0, t, f_test_latex, f_analytical)

# Test 2
#def f(y, x):
#    return x - y
#y0 = [0]
#f_analytical = t - 1 + np.exp(-t)
#f_test_latex="$y'+y=t , y(0)=1$ \n (Analytical: $y = t + e^{-x}$ - 1)"
#test(RK4_method, Euler_method, f, y0, t, f_test_latex, f_analytical)

## Double Pendulum - Solution ##

def double_pendulum(theta_1_0, theta_2_0, method, t_final):
    """Simulate double pendulum motion"""
    # System configuration
    m1, m2 = 1, 0.5  # masses [kg]
    L1, L2 = 1, 1.5  # lengths [m]
    g = 9.807        # gravity [m/s²]

    # Initial conditions
    omega_1_0, omega_2_0 = 0, 0  # initial angular velocities
    u0 = [theta_1_0, omega_1_0, theta_2_0, omega_2_0]

    # Time setup
    dt = 0.01  # Decreased time step for more precise crossings
    n_steps = round(t_final/dt)
    t = np.linspace(0, t_final, n_steps)
    
    def SDS(x, t, m1, m2, L1, L2, g):
        """Double Pendulum equations of motion"""
        dx = np.zeros(4)
        c = np.cos(x[0]-x[2])
        s = np.sin(x[0]-x[2])
        dx[0] = x[1]
        dx[2] = x[3]
        dx[1] = (m2*g*np.sin(x[2])*c - m2*s*(L1*c*x[1]**2 + L2*x[3]**2) - (m1+m2)*g*np.sin(x[0])) / (L1*(m1+m2*s**2))
        dx[3] = ((m1+m2)*(L1*x[1]**2*s - g*np.sin(x[2]) + g*np.sin(x[0])*c) + m2*L2*x[3]**2*s*c) / (L2*(m1+m2*s**2))
        return dx

    # Solve using specified method
    if method == "RK4":
        S = RK4_method(SDS, u0, t, args=(m1,m2,L1,L2,g))
    elif method == "Euler":
        S = Euler_method(SDS, u0, t, args=(m1,m2,L1,L2,g))
    else:
        raise ValueError("Method must be 'RK4' or 'Euler'")

    # Extract solutions
    theta1, omega1 = S[:,0], S[:,1]  # angle and angular velocity of mass 1
    theta2, omega2 = S[:,2], S[:,3]  # angle and angular velocity of mass 2

    # Convert to Cartesian coordinates
    x1 = L1*np.sin(theta1)
    y1 = -L1*np.cos(theta1)
    x2 = x1 + L2*np.sin(theta2)
    y2 = y1 - L2*np.cos(theta2)

    # Calculate energies
    U1 = m1 * g * y1  # potential energy of mass 1
    U2 = m2 * g * y2  # potential energy of mass 2
    K1 = 0.5 * m1 * (L1 * omega1)**2  # kinetic energy of mass 1
    K2 = 0.5 * m2 * ((L1 * omega1)**2 + (L2 * omega2)**2 + 
         2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))  # kinetic energy of mass 2
    
    return {
        'time': t,
        'theta1': theta1, 'omega1': omega1,
        'theta2': theta2, 'omega2': omega2,
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
        'U1': U1, 'U2': U2,
        'K1': K1, 'K2': K2
    }

def plot_trajectory(results):
    """Plot the trajectory of both pendulum masses"""
    plt.figure(figsize=(10, 10))
    plt.plot(results['x1'], results['y1'], 'r.', markersize=1, label='Mass 1')
    plt.plot(results['x2'], results['y2'], 'b.', markersize=1, label='Mass 2')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Pendulum Trajectory')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(results):
    """Plot phase space (angular position vs velocity) for both masses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(results['theta1'], results['omega1'], 'r-', linewidth=0.5)
    ax1.set_xlabel('θ₁ (rad)')
    ax1.set_ylabel('ω₁ (rad/s)')
    ax1.set_title('Phase Space - Mass 1')
    ax1.grid(True)
    
    ax2.plot(results['theta2'], results['omega2'], 'b-', linewidth=0.5)
    ax2.set_xlabel('θ₂ (rad)')
    ax2.set_ylabel('ω₂ (rad/s)')
    ax2.set_title('Phase Space - Mass 2')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_energy(results):
    """Plot energy evolution over time"""
    t = results['time']
    U1, U2 = results['U1'], results['U2']
    K1, K2 = results['K1'], results['K2']
    total_energy = U1 + U2 + K1 + K2
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, U1 + U2, 'r-', label='Potential Energy')
    plt.plot(t, K1 + K2, 'b-', label='Kinetic Energy')
    plt.plot(t, total_energy, 'k-', label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy Evolution')
    plt.grid(True)
    plt.legend()
    plt.show()

def animate_pendulum(results, save_gif=False):
    """Animate the double pendulum motion"""
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
    
    line1, = ax.plot([], [], 'o-', color='red', markersize=12, lw=2)    # mass 1
    line2, = ax.plot([], [], 'o-', color='blue', markersize=12, lw=2)   # mass 2
    time_template = 'Time = %.1f s'
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        return line1, line2, time_text
    
    def animate(i):
        x1 = [0, results['x1'][i]]
        y1 = [0, results['y1'][i]]
        x2 = [results['x1'][i], results['x2'][i]]
        y2 = [results['y1'][i], results['y2'][i]]
        
        line1.set_data(x1, y1)
        line2.set_data(x2, y2)
        time_text.set_text(time_template % results['time'][i])
        return line1, line2, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(results['time']), interval=20,
                                 blit=True)
    
    if save_gif:
        anim.save('double_pendulum.gif', writer='pillow')
    
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_poincare_section(results):
    """
    Calculate Poincaré section points from simulation results.
    The section is taken when theta1 crosses from negative to positive (y1 = 0).
    """
    theta1 = results['theta1']
    theta2 = results['theta2']
    omega2 = results['omega2']
    
    points_theta2 = []
    points_omega2 = []
    
    # Find crossings where theta1 goes from negative to positive
    for i in range(1, len(theta1)):
        if theta1[i-1] < 0 and theta1[i] > 0:
            # Linear interpolation to find exact crossing point
            t = -theta1[i-1] / (theta1[i] - theta1[i-1])
            # Interpolate theta2 and omega2 at crossing
            theta2_cross = theta2[i-1] + t * (theta2[i] - theta2[i-1])
            omega2_cross = omega2[i-1] + t * (omega2[i] - omega2[i-1])
            
            points_theta2.append(theta2_cross)
            points_omega2.append(omega2_cross)
    
    return np.array(points_theta2), np.array(points_omega2)

def plot_poincare_multiple(n_sims=10, t_final=200):
    """
    Generate and plot Poincaré section from multiple simulations with slightly different initial conditions.
    
    Args:
        n_sims: Number of simulations to run
        t_final: Duration of each simulation
    """
    all_theta2 = []
    all_omega2 = []
    
    # Base initial conditions
    base_theta1 = np.pi/4
    base_theta2 = np.pi/6
    
    # Run multiple simulations with slightly different initial conditions
    for i in range(n_sims):
        # Vary initial conditions slightly
        theta1_0 = base_theta1 + np.random.uniform(-0.1, 0.1)
        theta2_0 = base_theta2 + np.random.uniform(-0.1, 0.1)
        
        # Run simulation
        results = double_pendulum(theta1_0, theta2_0, "RK4", t_final)
        
        # Get Poincaré section points
        theta2, omega2 = calculate_poincare_section(results)
        
        all_theta2.extend(theta2)
        all_omega2.extend(omega2)
    
    # Plot combined results
    plt.figure(figsize=(10, 10))
    plt.plot(all_theta2, all_omega2, 'k.', ms=1)
    plt.xlabel('θ₂ (rad)')
    plt.ylabel('ω₂ (rad/s)')
    plt.title(f'Poincaré Section (θ₁ = 0, crossing positive)\n{n_sims} simulations, {t_final}s each')
    plt.grid(True)
    plt.show()

def main():
    """Example usage of the simulation and plotting functions"""
    # Run a single simulation for regular plots
    results = double_pendulum(theta_1_0=np.pi/4, theta_2_0=np.pi/6, 
                            method="RK4", t_final=50)
    
    # Generate regular plots
    plot_trajectory(results)
    plot_phase_space(results)
    plot_energy(results)
    
    # Generate enhanced Poincaré section with multiple simulations
    plot_poincare_multiple(n_sims=10, t_final=200)
    
    #animate_pendulum(results, save_gif=True) # this doesnt work

if __name__ == "__main__":
    main()