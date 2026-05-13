import random

import numpy as np


class PipeOptions:
    def __init__(self):
        self.conn_to_pipe = {
            frozenset({"N"}): "END_N",
            frozenset({"S"}): "END_S",
            frozenset({"E"}): "END_E",
            frozenset({"W"}): "END_W",
            frozenset({"N", "S"}): "I_0",
            frozenset({"E", "W"}): "I_90",
            frozenset({"N", "E"}): "L_0",
            frozenset({"E", "S"}): "L_90",
            frozenset({"S", "W"}): "L_180",
            frozenset({"N", "W"}): "L_270",
            frozenset({"N", "E", "W"}): "T_0",
            frozenset({"N", "E", "S"}): "T_90",
            frozenset({"E", "S", "W"}): "T_180",
            frozenset({"N", "S", "W"}): "T_270",
            frozenset({"N", "E", "S", "W"}): "X_0",
        }


def opposite(direction):
    return {"N": "S", "S": "N", "E": "W", "W": "E"}[direction]


class PipeGrid:
    def __init__(self, rows, cols, loop_prob=0.25):
        self.rows = rows
        self.cols = cols
        self.loop_prob = loop_prob
        self.connections = [[set() for _ in range(cols)] for _ in range(rows)]
        self._build_spanning_tree()
        self._add_loops()

    def _build_spanning_tree(self):
        visited = [[False] * self.cols for _ in range(self.rows)]

        def dfs(row, col):
            visited[row][col] = True
            directions = [("N", -1, 0), ("S", 1, 0), ("E", 0, 1), ("W", 0, -1)]
            random.shuffle(directions)
            for direction, dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and not visited[nr][nc]:
                    self.connections[row][col].add(direction)
                    self.connections[nr][nc].add(opposite(direction))
                    dfs(nr, nc)

        dfs(0, 0)

    def _add_loops(self):
        directions = [("N", -1, 0), ("S", 1, 0), ("E", 0, 1), ("W", 0, -1)]
        for row in range(self.rows):
            for col in range(self.cols):
                for direction, dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if (
                        0 <= nr < self.rows
                        and 0 <= nc < self.cols
                        and direction not in self.connections[row][col]
                        and random.random() < self.loop_prob
                    ):
                        self.connections[row][col].add(direction)
                        self.connections[nr][nc].add(opposite(direction))

    def to_pipe_ids(self, pipe_opts):
        return np.array(
            [
                [
                    pipe_opts.conn_to_pipe[frozenset(self.connections[row][col])]
                    for col in range(self.cols)
                ]
                for row in range(self.rows)
            ],
            dtype=object,
        )


class PipeVisualizerBW:
    def __init__(self, lanes=2, base=3):
        self.lanes = lanes
        self.base = base
        self.size = lanes * base
        self.patterns = self._make_patterns()

    def _make_patterns(self):
        size = self.size
        thickness = self.lanes
        mid = size // 2

        def empty():
            return np.zeros((size, size), dtype=int)

        def draw_cell(up, right, down, left):
            canvas = empty()
            lane_slice = slice(mid - thickness // 2, mid + (thickness + 1) // 2)
            if up:
                canvas[0 : mid + 1, lane_slice] = 1
            if down:
                canvas[mid:size, lane_slice] = 1
            if left:
                canvas[lane_slice, 0 : mid + 1] = 1
            if right:
                canvas[lane_slice, mid:size] = 1
            return canvas

        return {
            name: draw_cell(up, right, down, left)
            for name, up, right, down, left in [
                ("END_N", 1, 0, 0, 0),
                ("END_E", 0, 1, 0, 0),
                ("END_S", 0, 0, 1, 0),
                ("END_W", 0, 0, 0, 1),
                ("I_0", 1, 0, 1, 0),
                ("I_90", 0, 1, 0, 1),
                ("L_0", 1, 1, 0, 0),
                ("L_90", 0, 1, 1, 0),
                ("L_180", 0, 0, 1, 1),
                ("L_270", 1, 0, 0, 1),
                ("T_0", 1, 1, 0, 1),
                ("T_90", 1, 1, 1, 0),
                ("T_180", 0, 1, 1, 1),
                ("T_270", 1, 0, 1, 1),
                ("X_0", 1, 1, 1, 1),
            ]
        }

    def render(self, grid):
        rows, cols = grid.shape
        canvas = np.zeros((rows * self.size, cols * self.size), dtype=int)
        for row in range(rows):
            for col in range(cols):
                canvas[
                    row * self.size : (row + 1) * self.size,
                    col * self.size : (col + 1) * self.size,
                ] = self.patterns[grid[row, col]]
        return canvas

