# Minimax-AB_DepthLimitEval_BigTicTacToe
Minimax algorithm (with plans** to add Alpha-Beta pruning) and a depth-limited evaluation function.
Algorithm demo plays Tic Tac Toe on normal-size or 4x4 and larger boards. 

**I recently used this algorithm in C++ for a proprietary codebase and noted that A-B pruning was missing here.
I'll add it when there's time.

@game author: Piotr Szczurek, with AI functions implemented by Stephen Montague

Summary:

Big Tic-Tac-Toe using the Minimax algorithm with Alpha-Beta Pruning and a depth-limited evaluation function.
The 4x4 board and larger use the evaluation function to predict wins and avoid a loss or draw.
See further documentation for the Eval Function within the relevant code. 

Results:

In testing 10,000 games vs a random-play opponent:
AI (depth-limit 3) won over 92%, never lost, and drew less than 8% of games.
Heuristic could be further improved by new weights for end-game / extra-large boards.
Even so, this eval-method far outperforms others tested so far.

One small detail that significantly improves the AI, where WIN = 1, LOSS = -1:
In the terminal state of DRAW, return a near-zero random float, not zero, which
has the effect to randomize play when all options appear not to matter, versus a perfect opponent,
but might matter versus a flawed opponent, like a random-play bot or a tired human.
