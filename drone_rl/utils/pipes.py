import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize  # nearest-neighbor resize
import sys
sys.setrecursionlimit(10000)



# ---------------- PIPE DEFINITIONS ----------------

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

# ---------------- GRID GENERATION ----------------

class PipeGrid:
    def __init__(self, rows, cols, loop_prob=0.25):
        self.rows, self.cols, self.loop_prob = rows, cols, loop_prob
        self.connections = [[set() for _ in range(cols)] for _ in range(rows)]
        self._build_spanning_tree()
        self._add_loops()

    def _build_spanning_tree(self):
        visited = [[False]*self.cols for _ in range(self.rows)]
        def dfs(r, c):
            visited[r][c] = True
            dirs = [("N",-1,0), ("S",1,0), ("E",0,1), ("W",0,-1)]
            random.shuffle(dirs)
            for d, dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.rows and 0<=nc<self.cols and not visited[nr][nc]:
                    self.connections[r][c].add(d)
                    self.connections[nr][nc].add(opposite(d))
                    dfs(nr, nc)
        dfs(0,0)

    def _add_loops(self):
        for r in range(self.rows):
            for c in range(self.cols):
                for d, dr, dc in [("N",-1,0),("S",1,0),("E",0,1),("W",0,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<self.rows and 0<=nc<self.cols and d not in self.connections[r][c] and random.random() < self.loop_prob:
                        self.connections[r][c].add(d)
                        self.connections[nr][nc].add(opposite(d))

    def to_pipe_ids(self, pipe_opts):
        return np.array([[pipe_opts.conn_to_pipe[frozenset(self.connections[r][c])] for c in range(self.cols)] for r in range(self.rows)], dtype=object)

def opposite(d): return {"N":"S","S":"N","E":"W","W":"E"}[d]

class PipeVisualizerBW:
    def __init__(self, lanes=1, base=3):
        self.lanes, self.base = lanes, base
        self.s = lanes*base
        self.patterns = self._make_patterns()

    def _make_patterns(self):
        s, t, mid = self.s, self.lanes, self.s//2
        fill = lambda m,r0,r1,c0,c1: m.__setitem__(slice(r0,r1), np.append(m[r0:r1,:], np.zeros((r1-r0,c1-c0),dtype=int), axis=1)) or m
        empty = lambda: np.zeros((s,s), int)
        canvas = {}

        def draw_cell(up, right, down, left):
            m = empty()
            if up: m[0:mid+1, mid-t//2:mid+(t+1)//2]=1
            if down: m[mid:s, mid-t//2:mid+(t+1)//2]=1
            if left: m[mid-t//2:mid+(t+1)//2, 0:mid+1]=1
            if right: m[mid-t//2:mid+(t+1)//2, mid:s]=1
            return m

        for name, u,r,d,l in [
            ("END_N",1,0,0,0), ("END_E",0,1,0,0), ("END_S",0,0,1,0), ("END_W",0,0,0,1),
            ("I_0",1,0,1,0), ("I_90",0,1,0,1),
            ("L_0",1,1,0,0), ("L_90",0,1,1,0), ("L_180",0,0,1,1), ("L_270",1,0,0,1),
            ("T_0",1,1,0,1), ("T_90",1,1,1,0), ("T_180",0,1,1,1), ("T_270",1,0,1,1),
            ("X_0",1,1,1,1)
        ]: canvas[name] = draw_cell(u,r,d,l)
        return canvas

    def render(self, grid):
        rows, cols, s = grid.shape[0], grid.shape[1], self.s
        canvas = np.zeros((rows*s, cols*s), int)
        for r in range(rows):
            for c in range(cols):
                canvas[r*s:(r+1)*s, c*s:(c+1)*s] = self.patterns[grid[r,c]]
        return canvas

def vis_grid(grid_cw, lanes):
    fig, ax = plt.subplots(figsize=(8 * lanes, 8 * lanes))
    ax.imshow(grid_cw, cmap="gray", interpolation="nearest")

    # Get canvas dimensions
    h, w = grid_cw.shape

    # Pixel-level minor ticks (gridlines between every pixel)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)

    # Draw gray pixel gridlines
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)

    # Turn off major ticks completely
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig("fig.png")


# ---------------- MAIN ----------------

def main():
    grid_size = [20, 20]
    grid = PipeGrid(
        grid_size[0],
        grid_size[1], 
        loop_prob=0.5
    ).to_pipe_ids(PipeOptions())

    bw = PipeVisualizerBW().render(grid)

    lanes = 2
    bw = resize(
        bw, 
        (
            bw.shape[0]*lanes, 
            bw.shape[1]*lanes
        ), 
        order=0, 
        preserve_range=True, 
        anti_aliasing=False
    ).astype(int)

    vis_grid(bw, lanes)

if __name__ == "__main__":
    main()
