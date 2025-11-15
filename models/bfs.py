from collections import deque
from environment.maze import Cell # <-- Added import

def bfs_shortest_path_length(maze, start, goal):
    rows, cols = maze.shape
    visited = set()
    queue = deque([(start, 0)])  # (position, distance)
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return float('inf')  # no path

def _find_goal_cell(environment):
    """
    Finds the goal (exit) cell, which is the EMPTY cell in the maze
    that is not in the list of valid starting 'empty' cells.
    Based on ValueIterationModel._find_exit_cell().
    """
    for cell in environment.cells:
        if (environment.maze[cell[::-1]] == Cell.EMPTY and
                cell not in environment.empty):
            return cell[::-1]  # Return in (row, col) format for BFS
    raise RuntimeError("Goal (exit) cell could not be determined.")

def bfs_compute(environment, start_cell):
    # Convert (col, row) to (row, col) with ::-1 as stated in maze.py
    start = start_cell[::-1]
    
    # OLD: goal = environment.goal[::-1]
    # NEW: Find the goal programmatically
    goal = _find_goal_cell(environment)
    
    return bfs_shortest_path_length(environment.maze, start, goal)