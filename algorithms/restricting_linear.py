import numpy as np
import matplotlib as mpl

from blackbox_functions.zdt2 import ZDT2

# Use the pgf backend (must be set before pyplot imported)
# mpl.use('pgf')
import matplotlib.pyplot as plt
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.find_edge_points import FindEdgePoints
from paref.pareto_reflections.restrict_by_point import RestrictByPoint
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections

##################################
# Initialize Pareto reflections
reflection_1 = FindEdgePoints(dimension_domain=2, dimension=0)
reflection_2 = RestrictByPoint(restricting_point=np.array([0.7, 5]),
                               nadir=np.array([5, 5]))

reflection = ComposeReflections(reflection_2, reflection_1)
###################################
# Initialize blackbox function and MOO algorithm
function = ZDT2(input_dimensions=2)
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
        label='True PF'
        )

ax.plot(points.T[0],
        points.T[1],
        '^',
        color='black',
        label='Identified restricted area Pareto point')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.fill_between([0, 0.7], [5, 5], [0, 0], color='green', alpha=0.2, label='Restricted area')
ax.legend()
fig.show()

# fig.savefig('restricting_point.pgf', format='pgf')
