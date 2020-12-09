"""A module for homework 1."""
import random
import copy

import numpy as np

class EightPuzzleState:
    """A class for a state of an 8-puzzle game."""

    def __init__(self, board):
        """Create an 8-puzzle state."""
        self.action_space = {'u', 'd', 'l', 'r'}
        self.board = board
        for i, row in enumerate(self.board): #length of row board
            for j, v in enumerate(row): #length of column board
                if v == 0:
                    self.y = i
                    self.x = j

    
    def __repr__(self):
        """Return a string representation of a board."""
        output = []
        for row in self.board:
            row_string = ' | '.join([str(e) for e in row])
            output.append(row_string)
        return ('\n' + '-' * len(row_string) + '\n').join(output)

    def __str__(self):
        """Return a string representation of a board."""
        return self.__repr__()

    @staticmethod
    def initializeState():
        """
        Create an 8-puzzle state with a SHUFFLED tiles.
        
        Return
        ----------
        List[List[int]]
            A nested list containing integers representing numbers on a board
            e.g., [[0, 1, 2], [3, 4, 5], [6, 7, 8]] where 0 is a blank tile.
        """
        # TODO: 1
        board = np.arange(0, 9)
        np.random.shuffle(board)

        board = board.reshape((3, 3))
        return EightPuzzleState(board.tolist())

        # Testing isGoal method
        # return EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 0, 8]])


    def successor(self, action):
        """
        Move a blank tile in the current state, and return a new state.

        Parameters
        ----------
        action:  string 
            Either 'u', 'd', 'l', or 'r'.

        Return
        ----------
        EightPuzzleState or None
            A resulting 8-puzzle state after performing `action`.
            If the action is not possible, this method will return None.

        Raises
        ----------
        ValueError
            if the `action` is not in the action space
        
        """    
        if action not in self.action_space:
            raise ValueError(f'`action`: {action} is not valid.')
        # TODO: 2
        # YOU NEED TO COPY A BOARD BEFORE MODIFYING IT
        new_board = copy.deepcopy(self.board)

        """for i in range(0,3):
            for j in range(0,3):
                if new_board[i][j] is 0:
                    row_0 = i //self.y
                    column_0 = j //self.x"""

        isError = False
        success_i = self.y
        success_j = self.x

        if action is 'u':
            success_i -= 1
            isError = success_i < 0
        elif action is 'd':
            success_i += 1
            isError = success_i > 2
        elif action is 'l':
            success_j -= 1
            isError = success_j < 0
        elif action is 'r':
            success_j += 1
            isError = success_j > 2

        if isError:
            return None

        # TODO : Move 0
        swap = new_board[self.y][self.x]
        new_board[self.y][self.x] = new_board[success_i][success_j]
        new_board[success_i][success_j] = swap
        return EightPuzzleState(new_board)


    def is_goal(self, goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
        """
        Return True if the current state is a goal state.
        
        Parameters
        ----------
        goal_board (optional)
            The desired state of 8-puzzle.

        Return
        ----------
        Boolean
            True if the current state is a goal.
        
        """
        # TODO: 3

        return self.board == goal_board




class EightPuzzleNode:
    """A class for a node in a search tree of 8-puzzle state."""

    def __init__(
            self, state, parent=None, action=None, cost=1):
        """Create a node with a state."""
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        if parent is not None:
            self.path_cost = parent.path_cost + self.cost
        else:
            self.path_cost = 0

    def trace(self):
        """
        Return a path from the root to this node.

        Return
        ----------
        List[EightPuzzleNode]
            A list of nodes stating from the root node to the current node.

        """
        # TODO: 4
        current = self
        trace_list = []
        while current.parent is not None:
            trace_list.append(current)
            current = current.parent
        trace_list.append(current)
        return trace_list[::-1] #step from back to front 1 per time


def test_by_hand():
    """Run a CLI 8-puzzle game."""
    state = EightPuzzleState.initializeState()
    root_node = EightPuzzleNode(state, action='INIT')
    cur_node = root_node
    print(state)
    action = input('Please enter the next move (q to quit): ')
    while action != 'q':
        new_state = cur_node.state.successor(action)
        cur_node = EightPuzzleNode(new_state, cur_node, action)
        print(new_state)
        if new_state.is_goal():
            print('Congratuations!')
            break
        action = input('Please enter the next move (q to quit): ')

    print('Your actions are: ')
    for node in cur_node.trace():
        print(f'  - {node.action}')
    print(f'The total path cost is {cur_node.path_cost}')

if __name__ == '__main__':
    test_by_hand()