#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stephen Montague
4 Feb 2020
Spring 2020 Term 1
AI - 1: Machine Problem 2 - Minimax_AB + DL_Eval

EXTRA CREDIT VERSION (Implement Minimax with Optimizations)

Tic Tac Toe game created 18 Jan 2019.
AI updated on 9 Feb 2020 by student, changed from a random play bot to an optimal Minimax AI.

@game author: Piotr Szczurek, with AI functions implemented by Stephen Montague

Summary:

Tic-Tac-Toe using the Minimax algorithm with Alpha-Beta Pruning and a depth-limited evaluative function.
The 4x4 board and larger use the evaluation function to predict wins and avoid a loss or draw.
See further documentation for the Eval Function below. 

In testing 10,000 games vs a random-play opponent:
AI (depth-limit 3) won over 92%, never lost, and drew less than 8% of games.
Heuristic could be further improved by new weights for end-game / extra-large boards.
Even so, this eval-method far outperforms others tested so far.

One small detail that significantly improves the AI, where WIN = 1, LOSS = -1:
In the terminal state of DRAW, return a near-zero random float, not zero, which
has the effect to randomize play when all options appear not to matter, versus a perfect opponent,
but might matter versus a flawed opponent, like a random-play bot or a tired human.

"""

import numpy as np
import random


# self class is responsible for representing the game board
class GenGameBoard:

    # Constructor method - initializes board size and the marks of each position
    def __init__(self, size, marks):
        self.boardSize = size
        self.marks = marks
        self.marks[:, :] = ' '
        # Fields added by Stephen, depth limit is calibrated to perform 1 move / min or faster on a pc
        self.depthLimit = 3 if size < 6 else 2
        self.boardIsLarge = True if size > 3 else False

    # Prints the game board using current marks
    def printBoard(self):
        # Print column numbers
        print(' ', end='')
        for j in range(self.boardSize):
            print(" " + str(j + 1), end='')

        # Print rows with marks
        print("")
        for i in range(self.boardSize):
            # Print line separating the row
            print(" ", end='')
            for j in range(self.boardSize):
                print("--", end='')

            print("-")

            # Print row number
            print(i + 1, end='')

            # Print marks on self row
            for j in range(self.boardSize):
                print("|" + self.marks[i][j], end='')

            print("|")

        # Print line separating the last row
        print(" ", end='')
        for j in range(self.boardSize):
            print("--", end='')

        print("-")

    # Attempts to make a move given the row,col and mark
    # If move cannot be made, returns False and prints a message if mark is 'X'
    # Otherwise, returns True
    def makeMove(self, row, col, mark):
        possible = False  # Variable to hold the return value
        if row is None and col is None:
            return False

        # Change the row,col entries to array indexes
        row -= 1
        col -= 1

        if row < 0 or row >= self.boardSize or col < 0 or col >= self.boardSize:
            print("Not a valid row or column!")
            return False

        # Check row and col, and make sure space is empty
        # If empty, set the position to the mark and change possible to True
        if self.marks[row][col] == ' ':
            self.marks[row][col] = mark
            possible = True

        # Print out the message to the player if the move was not possible
        if not possible:
            print("\nPosition is already taken!")

        return possible

    # Determines whether a game winning condition exists
    # If so, returns True, and False otherwise
    def checkWin(self, mark):
        won = False  # Variable holding the return value

        # Check wins by examining each combination of positions

        # Check each row
        for i in range(self.boardSize):
            won = True
            for j in range(self.boardSize):
                if self.marks[i][j] != mark:
                    won = False
                    break
            if won:
                break

        # Check each column
        if not won:
            for i in range(self.boardSize):
                won = True
                for j in range(self.boardSize):
                    if self.marks[j][i] != mark:
                        won = False
                        break
                if won:
                    break

        # Check first diagonal
        if not won:
            for i in range(self.boardSize):
                won = True
                if self.marks[i][i] != mark:
                    won = False
                    break

        # Check second diagonal
        if not won:
            for i in range(self.boardSize):
                won = True
                if self.marks[self.boardSize - 1 - i][i] != mark:
                    won = False
                    break

        return won

    # Determines whether the board is full
    # If full, returns True, and False otherwise
    def noMoreMoves(self):
        return (self.marks != ' ').all()

    # TODO - These methods are ready for review.
    def makeCompMove(self):
        action = self.alphaBetaSearch()  # Returns int tuple of best action
        self.makeMove(*action, 'O')
        print("Computer chose:", action)

    def alphaBetaSearch(self):  # Returns best value action coordinate
        possibleActions = self.genPossibleActions()
        for action in possibleActions:  # For each action, act, compute value, add value to dict, undo act
            self.makeMove(*action, 'O')
            possibleActions[action] = self.maxValue()
            self.undoMove(*action)
        return max(possibleActions, key=possibleActions.get)

    def maxValue(self, depth=0, alpha=-2, beta=2):
        if self.checkWin('O'):
            return COMPUTER_WON
        elif self.noMoreMoves():
            return random.uniform(-0.1, 0.0)
        else:
            depth += 1

            if self.boardIsLarge and depth > self.depthLimit:
                minimizer = "human"
                return self.evaluateBoard(minimizer)

            possibleActions = self.genPossibleActions()
            for action in possibleActions:
                self.makeMove(*action, 'X')
                possibleActions[action] = self.minValue(depth, alpha, beta)
                self.undoMove(*action)
                if alpha >= beta:
                    break
            return max(alpha, possibleActions[min(possibleActions, key=possibleActions.get)])

    def minValue(self, depth=0, alpha=-2, beta=2):
        if self.checkWin('X'):
            return PLAYER_WON
        elif self.noMoreMoves():
            return random.uniform(0, 0.1)
        else:
            depth += 1

            if self.boardIsLarge and depth > self.depthLimit:
                maximizer = "computer"
                return self.evaluateBoard(maximizer)

            possibleActions = self.genPossibleActions()
            for action in possibleActions:
                self.makeMove(*action, 'O')
                possibleActions[action] = self.maxValue(depth, alpha, beta)
                self.undoMove(*action)
                if beta <= alpha:
                    break
            return min(beta, possibleActions[max(possibleActions, key=possibleActions.get)])

    def genPossibleActions(self):  # Build dictionary of action coordinate keys w/ neutral value
        possibleActions = {}
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.marks[i, j] == ' ':
                    possibleActions[(i + 1, j + 1)] = 0  # Add to dict, convert index to coordinate
        return possibleActions

    def undoMove(self, row, col):
        self.marks[row-1][col-1] = ' '  # Convert coordinate to index

    def evaluateBoard(self, player):
        """ Returns float between (-1, 1) from the percent of lines open to a win + corner weight + random modifier

                percentOpen = (worstCase - sumLinesBlocked) / worstCase

                evaluation = (percentOpen + corner) + random

            worstCase := max blocked wins on any board, gameBoard.size * 2 + 2
            sumLinesBlocked := the sum of rows, columns, and diagonals blocked by opponent
            corner := plus 5 percent of percentOpen per corner occupied by player
            random := plus or minus up to 1 percent of formula

            In theory, the computer can find a move that seems more useful, that is, less blocked,
            while prioritizing the center and corners, randomizing play when all moves seem equal. """

        players = {"human": 'X', "computer": 'O'}

        if player == "human":
            player_mark = players["human"]
            opponent_mark = players["computer"]
        else:
            player_mark = players["computer"]
            opponent_mark = players["human"]

        sumLinesBlocked = 0
        worstCase = (self.boardSize * 2) + 2

        # Add rows blocked
        for row in range(self.boardSize):
            if opponent_mark in self.marks[row]:
                sumLinesBlocked += 1

        # Add columns blocked
        for col in range(self.boardSize):
            if np.any(self.marks[:, col] == opponent_mark):
                sumLinesBlocked += 1

        # Add left diagonal if blocked
        for i in range(self.boardSize):
            if self.marks[i, i] == opponent_mark:
                sumLinesBlocked += 1
                break

        # Add right diagonal if blocked
        for i in range(self.boardSize):
            if self.marks[self.boardSize - i - 1, i] == opponent_mark:
                sumLinesBlocked += 1
                break

        # Ensure valid, percent open should not be <= 0
        if sumLinesBlocked >= worstCase:
            sumLinesBlocked = worstCase - 1

        # Calculate formula
        percentOpen = float(worstCase - sumLinesBlocked) / worstCase

        # Add weight per corner
        corners = [self.marks[0, 0], self.marks[0, self.boardSize-1],
                   self.marks[self.boardSize-1, 0], self.marks[self.boardSize-1, self.boardSize-1]]
        cornerWeight = percentOpen * 0.1  # Weight 10%
        for corner in corners:
            if corner == player_mark:
                percentOpen = percentOpen + cornerWeight

        # Add random
        randomVariance = percentOpen * (random.randint(-5, 5)/100.0)  # Random up to +/- 5%

        # Make evaluation
        if player == "computer":
            evaluation = percentOpen + randomVariance
        else:  # Player is human, minimizer, flip eval
            evaluation = -1 * (percentOpen + randomVariance)

        return evaluation

    def makeHumanMove(self):  # For testing - auto pilot the human
        possibleHumanActions = list(self.genPossibleActions())
        row, col = random.choice(possibleHumanActions)
        print("Mr. Jim Bob chose: "+str(row)+","+str(col))
        return row, col


# Print out the header info
print("EXTRA CREDIT VERSION")
print("CLASS: Artificial Intelligence, Lewis University")
print("NAME: Stephen Montague")

COMPUTER_WON = 1  # Names & values changed (by Stephen) to use with minimax
PLAYER_WON = -1
DRAW = 0
wrongInput = False
boardSize = int(input("Please enter the size of the board n (e.g. n=3,4,5,...): "))

# Create the game board of the given size
board = GenGameBoard(boardSize, np.empty((boardSize, boardSize), dtype='str'))

board.printBoard()  # Print the board before starting the game loop

# Game loop
while True:
    # *** Player's move ***

    # Try to make the move and check if it was possible
    # If not possible get col,row inputs from player
    row_, column = None, None
    while not board.makeMove(row_, column, 'X'):
        print("Player's Move")
        row_, column = input("Choose your move (row, column): ").split(',')  # For auto-pilot, use board.makeHumanMove()
        row_ = int(row_)
        column = int(column)


    # Display the board again
    board.printBoard()

    # Check for ending condition
    # If game is over, check if player won and end the game
    if board.checkWin('X'):
        # Player won
        result = PLAYER_WON
        break
    elif board.noMoreMoves():
        # No moves left -> draw
        result = DRAW
        break

    # *** Computer's move ***
    board.makeCompMove()

    # Print out the board again
    board.printBoard()

    # Check for ending condition
    # If game is over, check if computer won and end the game
    if board.checkWin('O'):
        # Computer won
        result = COMPUTER_WON
        break
    elif board.noMoreMoves():
        # No moves left -> draw
        result = DRAW
        break

# Check the game result and print out the appropriate message
print("GAME OVER")
if result == PLAYER_WON:
    print("You Win!")
elif result == COMPUTER_WON:
    print("You Lose!")
else:
    print("It's a draw!")
