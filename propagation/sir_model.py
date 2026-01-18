"""
SIR Model for Misinformation Propagation
Simulates the spread of misinformation using epidemiological model
S = Susceptible, I = Infected, R = Recovered
"""

import numpy as np
from scipy.integrate import odeint

class SIRModel:
    def __init__(self, population=10000, initial_infected=10, initial_recovered=0):
        """
        Initialize SIR model
        
        Args:
            population: Total population size
            initial_infected: Initial number of infected individuals
            initial_recovered: Initial number of recovered individuals
        """
        self.N = population
        self.I0 = initial_infected
        self.R0 = initial_recovered
        self.S0 = population - initial_infected - initial_recovered
        
    def sir_equations(self, y, t, beta, gamma):
        """
        SIR differential equations
        
        Args:
            y: Current state [S, I, R]
            t: Time
            beta: Infection rate (contact rate * transmission probability)
            gamma: Recovery rate (1 / infectious period)
            
        Returns:
            Derivatives [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y
        N = self.N
        
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        
        return [dSdt, dIdt, dRdt]
    
    def simulate(self, days=80, infection_rate=0.5, recovery_rate=0.1):
        """
        Simulate SIR model over time
        
        Args:
            days: Number of days to simulate
            infection_rate: Beta parameter (how easily misinformation spreads)
            recovery_rate: Gamma parameter (how quickly people stop sharing)
            
        Returns:
            dict: Time series data for S, I, R populations
        """
        # Initial conditions
        y0 = [self.S0, self.I0, self.R0]
        
        # Time vector
        t = np.linspace(0, days, days)
        
        # Solve ODE
        solution = odeint(self.sir_equations, y0, t, args=(infection_rate, recovery_rate))
        
        S, I, R = solution.T
        
        # Calculate additional metrics
        peak_infected = int(np.max(I))
        peak_day = int(np.argmax(I))
        final_recovered = int(R[-1])
        total_reached = final_recovered  # People who were exposed
        
        return {
            'time': t.tolist(),
            'susceptible': S.tolist(),
            'infected': I.tolist(),
            'recovered': R.tolist(),
            'metrics': {
                'peak_infected': peak_infected,
                'peak_day': peak_day,
                'total_reached': total_reached,
                'reach_percentage': (total_reached / self.N) * 100,
                'infection_rate': infection_rate,
                'recovery_rate': recovery_rate
            }
        }
    
    def calculate_r0(self, infection_rate, recovery_rate):
        """
        Calculate basic reproduction number (R0)
        R0 = β / γ
        If R0 > 1, misinformation will spread exponentially
        If R0 < 1, misinformation will die out
        
        Args:
            infection_rate: Beta parameter
            recovery_rate: Gamma parameter
            
        Returns:
            float: R0 value
        """
        return infection_rate / recovery_rate
    
    def predict_spread_severity(self, is_fake=True):
        """
        Predict spread parameters based on news type
        Fake news typically spreads faster (higher beta)
        
        Args:
            is_fake: Whether the news is fake
            
        Returns:
            dict: Simulation parameters and results
        """
        if is_fake:
            # Fake news spreads faster and people share longer
            beta = 0.5  # High infection rate
            gamma = 0.1  # Low recovery rate
            severity = "High"
        else:
            # Real news spreads slower
            beta = 0.3  # Lower infection rate
            gamma = 0.15  # Higher recovery rate
            severity = "Moderate"
        
        r0 = self.calculate_r0(beta, gamma)
        simulation = self.simulate(infection_rate=beta, recovery_rate=gamma)
        
        return {
            'severity': severity,
            'r0': r0,
            'interpretation': self._interpret_r0(r0),
            'simulation': simulation
        }
    
    def _interpret_r0(self, r0):
        """Interpret R0 value"""
        if r0 > 2:
            return "Highly contagious - will spread rapidly"
        elif r0 > 1:
            return "Contagious - will spread but slowly"
        else:
            return "Not contagious - will die out naturally"


# For quick testing
if __name__ == "__main__":
    sir = SIRModel(population=10000, initial_infected=10)
    
    print("SIR Model Simulation for Fake News:")
    result = sir.predict_spread_severity(is_fake=True)
    print(f"Severity: {result['severity']}")
    print(f"R0: {result['r0']:.2f}")
    print(f"Interpretation: {result['interpretation']}")
    
    metrics = result['simulation']['metrics']
    print(f"\nPeak Infected: {metrics['peak_infected']} people on day {metrics['peak_day']}")
    print(f"Total Reached: {metrics['total_reached']} ({metrics['reach_percentage']:.1f}%)")
    
    print("\n" + "="*50)
    print("SIR Model Simulation for Real News:")
    result = sir.predict_spread_severity(is_fake=False)
    print(f"Severity: {result['severity']}")
    print(f"R0: {result['r0']:.2f}")
    print(f"Interpretation: {result['interpretation']}")
    
    metrics = result['simulation']['metrics']
    print(f"\nPeak Infected: {metrics['peak_infected']} people on day {metrics['peak_day']}")
    print(f"Total Reached: {metrics['total_reached']} ({metrics['reach_percentage']:.1f}%)")
