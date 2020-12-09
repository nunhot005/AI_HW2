"""A module for homework 2. Version 3."""
# noqa: D413

import abc
import heapq
import itertools
from collections import defaultdict
from hw1 import EightPuzzleState, EightPuzzleNode

REMOVED = '<removed-task>'  # placeholder for a removed task


def eightPuzzleH1(state, goal_state: EightPuzzleState = EightPuzzleState(([1, 2, 3], [4, 5, 6], [7, 8, 0]))):
    """
    Return the number of misplaced tiles including blank.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    # TODO 1:
    # [0,0 0,1 0,2]
    # [1,0 1,1 1,2]
    # [2,0 2,1 2,2]
    countFalse = 0
    for i in range(3):
        for j in range(3):
            if state.board[i][j] != goal_state.board[i][j]:
                countFalse += 1
    return countFalse


def eightPuzzleH2(state, goal_state=EightPuzzleState(([1, 2, 3], [4, 5, 6], [7, 8, 0]))):
    """
    Return the total Manhattan distance from goal position of all tiles.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    # TODO 2:
    manhattan = 0
    for i in range(3):
        for j in range(3):
            for x in range(3):
                for y in range(3):
                    if state.board[i][j] == goal_state.board[x][y]:
                        current_index = abs(i-x) + abs(j-y)
                        manhattan += current_index
    return manhattan


class Frontier(abc.ABC):
    """An abstract class of a frontier."""

    def __init__(self):
        """Create a frontier."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self):
        """Return True if empty."""
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        raise NotImplementedError()


class DFSFrontier(Frontier):
    """An example of how to implement a depth-first frontier (stack)."""

    def __init__(self):
        """Create a frontier."""
        self.stack = []

    def is_empty(self):
        """Return True if empty."""
        return len(self.stack) == 0

    def add(self, node):
        """
        Add a node into the frontier.
        
        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        for n in self.stack:
            # HERE we assume that state implements __eq__() function.
            # This could be improve further by using a set() datastructure,
            # by implementing __hash__() function.
            if n.state == node.state:
                return None
        self.stack.append(node)

    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        return self.stack.pop()


class GreedyFrontier(Frontier):
    """A frontier for greedy search."""

    def is_empty(self):
        return len(self.pq) == 0

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state, goal_state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()

        """
        self.h = h_func
        self.goal = goal_state
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = itertools.count()  # unique sequence count

        # TODO: 3
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.

    def add(self, node):
        'Add a new node or update the priority of an existing node'
        if node in self.entry_finder:
            self.remove_node(node)
        count = next(self.counter)
        entry = [self.h(node.state, self.goal), count, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.pq, entry)

    def remove_node(self, node):
        'Mark an existing node as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(node)
        entry[-1] = REMOVED

    def next(self):
        'Remove and return the lowest priority node. Raise KeyError if empty.'
        while self.pq:
            priority, count, node = heapq.heappop(self.pq)
            if node is not REMOVED:
                del self.entry_finder[node]
                return node
        raise KeyError('pop from an empty priority queue')


class AStarFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()


        """
        self.h = h_func
        self.goal = goal_state
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = itertools.count()  # unique sequence count

        # TODO: 4
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.

    def is_empty(self):
        return len(self.pq) == 0

    def add(self, node):
        'Add a new node or update the priority of an existing node'
        if node in self.entry_finder:
            self.remove_node(node)
        count = next(self.counter)
        entry = [self.h(node.state, self.goal) + node.cost, count, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.pq, entry)

    def remove_node(self, node):
        'Mark an existing node as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(node)
        entry[-1] = REMOVED

    def next(self):
        'Remove and return the lowest priority node. Raise KeyError if empty.'
        while self.pq:
            priority, count, node = heapq.heappop(self.pq)
            if node is not REMOVED:
                del self.entry_finder[node]
                return node
        raise KeyError('pop from an empty priority queue')


def _parity(board):
    """Return parity of a square matrix."""
    inversions = 0
    nums = []
    for row in board:
        for value in row:
            nums.append(value)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] != 0 and nums[j] != 0 and nums[i] > nums[j]:
                inversions += 1
    return inversions % 2


def _is_reachable(board1, board2):
    """Return True if two N-Puzzle state are reachable to each other."""
    return _parity(board1) == _parity(board2)


def graph_search(init_state, goal_state, frontier):
    """
    Search for a plan to solve problem.

    Parameters
    ----------
    init_state : EightPuzzleState
        an initial state
    goal_state : EightPuzzleState
        a goal state
    frontier : Frontier
        an implementation of a frontier which dictates the order of exploreation.
    
    Returns
    ----------
    plan : List[string] or None
        A list of actions to reach the goal, None if the search fails.
        Your plan should NOT include 'INIT'.
    num_nodes: int
        A number of nodes generated in the search.

    """
    if not _is_reachable(init_state.board, goal_state.board):
        return None, 0
    if init_state.is_goal(goal_state.board):
        return [], 0
    num_nodes = 0
    solution = []
    # Perform graph search
    root_node = EightPuzzleNode(init_state, action='INIT')
    frontier.add(root_node)
    num_nodes += 1

    # TODO: 5
    explore = set()
    while not frontier.is_empty():
        currNode = frontier.next()
        if currNode.state.is_goal():
            break

        num_nodes += 1

        if currNode.state not in explore:
            explore.add(currNode.state)
            children = get_leaf_node(currNode)
            for c in children:
                frontier.add(c)

    return solution, num_nodes

def get_leaf_node(node: EightPuzzleNode):
    leafs: list[EightPuzzleNode] = []

    state: EightPuzzleState = node.state

    allPossibleAct = {'l', 'u', 'r', 'd'}
    if state.y - 1 < 0:
        allPossibleAct.remove('u')
    if state.y + 1 > 2:
        allPossibleAct.remove('d')
    if state.x - 1 < 0:
        allPossibleAct.remove('l')
    if state.x + 1 > 2:
        allPossibleAct.remove('r')

    for action in allPossibleAct:
        movedState = state.successor(action)
        leafs.append(EightPuzzleNode(movedState, node, action))

    return leafs


def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()
    while not _is_reachable(goal_state.board, init_state.board):
        init_state = EightPuzzleState.initializeState()

    #frontier = DFSFrontier() # Change this to your own implementation.
    #frontier = GreedyFrontier(eightPuzzleH1, goal_state)  # Change this to your own implementation.
    #frontier = GreedyFrontier(eightPuzzleH2, goal_state)  # Change this to your own implementation.
    #frontier = AStarFrontier(eightPuzzleH1, goal_state)  # Change this to your own implementation.
    frontier = AStarFrontier(eightPuzzleH2, goal_state)  # Change this to your own implementation.

    if verbose:
        print(init_state)
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    if verbose:
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose:
        for action in plan:
            print(f'- {action}')
    return len(plan), num_nodes


def experiment(n=10000):
    """Run experiments and report number of nodes generated."""
    result = defaultdict(list)
    for __ in range(n):
        d, n = test_by_hand(True)
        result[d].append(n)
    max_d = max(result.keys())
    for i in range(max_d + 1):
        n = result[d]
        if len(n) == 0:
            continue
        print(f'{d}, {len(n)}, {sum(n) / len(n)}')


if __name__ == '__main__':
    __, __ = test_by_hand()
    #experiment()  #  run graph search 10000 times and report result.
