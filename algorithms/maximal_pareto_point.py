import numpy as np
import matplotlib as mpl

from blackbox_functions.zdt1 import ZDT1
# Use the pgf backend (must be set before pyplot imported)
# mpl.use('pgf')
import matplotlib.pyplot as plt

from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia

##################################
# Initialize Pareto reflection corresponding to some maximal Pareto point

reflection = MinimizeWeightedNormToUtopia(utopia_point=np.array([0, 0]),
                                          potency=2,
                                          scalar=np.ones(2),
                                          )

###################################
# Initialize blackbox function and MOO algorithm
function = ZDT1(input_dimensions=2)
moo = DifferentialEvolutionMinimizer()
stopping_criteria = MaxIterationsReached(max_iterations=2)
moo.apply_to_sequence(blackbox_function=function,
                      sequence_pareto_reflections=reflection,
                      stopping_criteria=stopping_criteria)

points = np.array([evaluation[1] for evaluation in function.evaluations])

line = function.return_true_pareto_front()

##################################
# Plot results
fig, ax = plt.subplots()

ax.plot(line.T[0],
        line.T[1],
        ':',
        color='gray',
        label='True Pareto front'
        )

ax.plot(points.T[0],
        points.T[1],
        '^',
        color='black',
        label='Identified maximal Pareto point')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.legend()
fig.show()

# fig.savefig('maximal_pp.pgf', format='pgf')
