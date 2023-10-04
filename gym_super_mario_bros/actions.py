"""Static action sets for binary to discrete action space wrappers."""


# actions for the simple run right environment
RIGHT_ONLY = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'], # Jump and run right
]


# actions for very simple movement
# A is to jump
# B is to run
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'], # Walk right
    ['right', 'A'], # Jump right 
    ['right', 'B'], # Run right
    ['right', 'A', 'B'], # Jump and run right
    ['A'], # Jump only
    ['left'], # Walk left
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]
