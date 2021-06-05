# This is a copy of the notebook "Alternative game visualization"
# https://www.kaggle.com/egrehbbt/alternative-game-visualization

import numpy as np
from matplotlib import animation, pyplot as plt

NUM_ROWS, NUM_COLUMNS = 7, 11


def flat_to_point(_x):
    assert 0 <= _x < NUM_COLUMNS * NUM_ROWS
    return _x % NUM_COLUMNS, NUM_ROWS - _x // NUM_COLUMNS - 1


class GameAnimation:
    ACTION_TO_SIGN = {"NORTH": "↑", "EAST": "→", "SOUTH": "↓", "WEST": "←"}
    GOOSE_ID_TO_COLOR = {0: "white", 1: "blue", 2: "green", 3: "red"}
    FOOD_COLOR = "gold"

    def __init__(self, ax, steps, **kwargs):
        self.steps = steps
        self.num_geese = len(steps[0][0]["observation"]["geese"])
        self.plot_objects = self.init_func(ax, **kwargs)

    @staticmethod
    def goose_score(state, goose_id: int):
        return state[goose_id]["reward"]

    @staticmethod
    def goose_action(steps, step, goose_id):
        if step + 1 >= len(steps):
            return

        if steps[step][goose_id]["status"] != "ACTIVE":
            return

        return steps[step + 1][goose_id]["action"]

    @staticmethod
    def food_positions(state):
        points = state[0]["observation"]["food"]
        points = [flat_to_point(x) for x in points]
        assert len(points) == 2
        x, y = [x[0] for x in points], [x[1] for x in points]
        if x[0] == x[1] and y[0] == y[1]:
            x[0] -= 0.25
            x[1] += 0.25
        return x, y

    @staticmethod
    def player_positions(state, goose_id: int):
        points = state[0]["observation"]["geese"][goose_id]
        points = [flat_to_point(x) for x in points]

        if len(points) == 0:
            return [], []

        if len(points) == 1:
            return [points[0][0]], [points[0][1]]

        last_x, last_y = points[0][0], points[0][1]
        xx, yy = [last_x], [last_y]
        for p in points[1:]:
            x, y = p[0], p[1]

            if x == 0 and last_x == NUM_COLUMNS - 1:
                xx += [NUM_COLUMNS + 2, NUM_COLUMNS + 2, -2, -2]
                yy += [y, -2, -2, y]

            if x == NUM_COLUMNS - 1 and last_x == 0:
                xx += [-2, -2, NUM_COLUMNS + 2, NUM_COLUMNS + 2]
                yy += [y, -2, -2, y]

            if y == 0 and last_y == NUM_ROWS - 1:
                xx += [x, -2, -2, x]
                yy += [NUM_ROWS + 2, NUM_ROWS + 2, -2, -2]

            if y == NUM_ROWS - 1 and last_y == 0:
                xx += [x, -2, -2, x]
                yy += [-2, -2, NUM_ROWS + 2, NUM_ROWS + 2]

            xx.append(x)
            yy.append(y)
            last_x, last_y = x, y

        return xx, yy

    @staticmethod
    def head_positions(state, goose_id: int):
        points = state[0]["observation"]["geese"][goose_id]
        points = [flat_to_point(x) for x in points]
        return [x[0] for x in points[:1]], [x[1] for x in points[:1]]

    def init_func(self, ax, padding=0.5):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(-0.5 - padding, NUM_ROWS - 0.5 + padding)
        ax.set_xlim(-0.5 - padding, NUM_COLUMNS - 0.5 + padding)

        # GRID
        for x in np.arange(-0.5, NUM_ROWS + 1.5, 1):
            ax.plot([-0.5, NUM_COLUMNS - 0.5], [x, x], color="black")

        for x in np.arange(-0.5, NUM_COLUMNS + 1.5, 1):
            ax.plot([x, x], [-0.5, NUM_ROWS - 0.5], color="black")

        # FOOD
        food, *_ = ax.plot([], [], "*", color=self.FOOD_COLOR, ms=24)

        # GEESE
        geese = []
        heads = []
        for i in range(self.num_geese):
            color = self.GOOSE_ID_TO_COLOR[i]
            g, *_ = ax.plot([], [], color=color, alpha=0.7)
            g.set_linewidth(10)
            h, *_ = ax.plot([], [], "D", color=color, alpha=0.7, ms=14)
            geese.append(g)
            heads.append(h)

        # TEXT
        step_text = ax.text(-0.5, NUM_ROWS, "", fontsize=16)
        scores = []
        for i in range(self.num_geese):
            color = self.GOOSE_ID_TO_COLOR[i]
            score_text = ax.text(
                NUM_COLUMNS,
                NUM_ROWS - i,
                "",
                fontsize=20,
                color=color,
                family="monospace",
            )
            scores.append(score_text)

        return food, geese, heads, step_text, scores

    def animate(self, step):
        food, geese, heads, step_text, scores = self.plot_objects

        state = self.steps[step]

        # FOOD
        food.set_data(self.food_positions(state))

        # GEESE
        for i in range(self.num_geese):
            geese[i].set_data(self.player_positions(state, i))
            heads[i].set_data(self.head_positions(state, i))

        # TEXT
        step_text.set_text(f"Step {step}")
        for i in range(self.num_geese):
            reward = self.goose_score(state, i)
            action = self.goose_action(self.steps, step, i)
            sign = self.ACTION_TO_SIGN.get(action, " ")
            scores[i].set_text(f"{sign} {reward}")

        return (food, *geese, *heads, step_text, *scores)


def animate(env, width=9, height=6, padding=0.5):
    fig = plt.figure(figsize=(width, height), facecolor=(0.5, 0.7, 0.4))
    ax = plt.axes((0.05, 0.1, 0.75, 0.75))
    plt.axis("off")

    game_animation = GameAnimation(ax, env.steps, padding=padding)

    return animation.FuncAnimation(
        fig,
        func=game_animation.animate,
        interval=400,
        blit=True,
        repeat=False,
        save_count=len(env.steps),
    )
