# Four in a row (connect four) bot changelog

- v1: testbot, does nothing but randomly select a column to play
- v2:  first actual bot. minmax with alpha-beta pruning, iterative deepening, and time management. Slight preference for center columns over outer columns (weights [0,1,2,3,2,1,0])
- v3: changed column preference to center, then corners, then edges (weights [1,0,0,2,0,0,1)
- v4: output error
- v5:
    - switched back to added weights of [0,1,2,3,2,1,0]
    - added even/odd threats (player 1 wants odd threats, player 2 wants even threats) with a multiplier of 1.5
    - fixed one diagonal not being evaluated
    - when predicting loss, now tries to delay loss (used to just pick a random move)
- v6: fixed that it didn't detect a win condition in two diagonals
- v7: fixed bug where, if it had only losing options, it would always pick column 0 even if that wasn't a valid move. Also commented out unnecessary move simulation.
- v8: added transposition table with max of 1e7 elements. On my lowly laptop, it seems to tank performance: only 1/3 as many nodes evaluated, but let's see what theaigames.com's beefy machine does.

# To-do

- Layer alpha-beta on top of iterative deepening (i.e. reorder the nodes so alpha-beta starts with the max-value one)
- hash table of previously evaluated boards