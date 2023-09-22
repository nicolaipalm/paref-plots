import numpy as np
import matplotlib as mpl

# Use the pgf backend (must be set before pyplot imported)
#mpl.use('pgf')
import matplotlib.pyplot as plt

from examples.blackbox_functions.two_dimensional.zdt2 import ZDT2
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.fill_gap import FillGap

##################################

function = ZDT2(input_dimensions=2)
gap_points = np.array([function.return_true_pareto_front()[30],
                       function.return_true_pareto_front()[80]])

# calculate center
# center = np.mean(gap_points, axis=1) doesnt work
center = 1 / 2 * (gap_points[0] + gap_points[1])



##################################
reflection = FillGap(gap_points=gap_points,dimension_domain=2)

###################################
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
        label='True Pareto front'
        )

ax.plot(points.T[0],
        points.T[1],
        '^',
        color='black',
        label='Identified gap fill Pareto point')

ax.plot(gap_points.T[0],
        gap_points.T[1],
        'o',
        color='green',
        label='Gap points')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.legend()
fig.show()
#fig.savefig('fill_gap.pgf', format='pgf')
