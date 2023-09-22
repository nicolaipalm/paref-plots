import numpy as np
import matplotlib as mpl
# Use the pgf backend (must be set before pyplot imported)
mpl.use('pgf')
import matplotlib.pyplot as plt
from examples.blackbox_functions.two_dimensional.zdt1 import ZDT1
from examples.blackbox_functions.two_dimensional.zdt2 import ZDT2
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.find_edge_points import FindEdgePoints

##################################
reflection = FindEdgePoints(dimension_domain=2, dimension=0)

###################################
function = ZDT1(input_dimensions=2)
moo = DifferentialEvolutionMinimizer()
stopping_criteria = MaxIterationsReached(max_iterations=2)
moo.apply_to_sequence(blackbox_function=function,
                      sequence_pareto_reflections=reflection,
                      stopping_criteria=stopping_criteria)

points = np.array([evaluation[1] for evaluation in function.evaluations])

line = function.return_true_pareto_front()

##################################
# plot
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
        label='Identified 1-Pareto point')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.legend()
fig.show()

fig.savefig('edge_point.pgf', format='pgf')