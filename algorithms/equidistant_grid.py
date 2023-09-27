import numpy as np
import matplotlib as mpl

from blackbox_functions.zdt2 import ZDT2

# Use the pgf backend (must be set before pyplot imported)
# mpl.use('pgf')
import matplotlib.pyplot as plt

from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.multi_dimensional.fill_gaps_of_pareto_front_sequence import \
    FillGapsOfParetoFrontSequence

##################################
# Initialize blackbox function and evaluate (manually) at edge points

function = ZDT2(input_dimensions=2)

#
function(np.array([0, 0]))
function(np.array([1, 0]))

##################################
# Initialize Pareto reflection iteratively filling gaps of (found) Pareto front
reflection = FillGapsOfParetoFrontSequence()

###################################
# Initialize MOO algorithm
moo = DifferentialEvolutionMinimizer()
stopping_criteria = MaxIterationsReached(max_iterations=7)
moo.apply_to_sequence(blackbox_function=function,
                      sequence_pareto_reflections=reflection,
                      stopping_criteria=stopping_criteria)

points = function.y

line = function.return_true_pareto_front()

##################################
# Plot result
fig, ax = plt.subplots(figsize=(6, 6))

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
        label='Identified equidistant grid of Pareto points')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.legend()
fig.show()
# fig.savefig('equidistant_grid.pgf', format='pgf')
