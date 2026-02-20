FIGURES = {
    "I": {"id": 1, "shape": [[1, 5, 9, 13], [4, 5, 6, 7]], "color": (0, 240, 255)},
    "S": {"id": 2, "shape": [[4, 5, 9, 10], [2, 6, 5, 9]], "color": (57, 255, 20)},
    "Z": {"id": 3, "shape": [[6, 7, 9, 10], [1, 5, 6, 10]], "color": (255, 0, 85)},
    "L": {
        "id": 4,
        "shape": [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        "color": (255, 120, 0),
    },
    "J": {
        "id": 5,
        "shape": [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        "color": (20, 80, 255),
    },
    "T": {
        "id": 6,
        "shape": [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        "color": (170, 0, 255),
    },
    "O": {"id": 7, "shape": [[1, 2, 5, 6]], "color": (255, 235, 0)},
}


class TetrisPiece:
    def __init__(self, figure, coordinates):
        self.shapes = figure["shape"]
        self.active_shape = self.shapes[0]

        self.id = figure["id"]
        self.color = figure["color"]

        self.x = coordinates[0]
        self.y = coordinates[1]
        self.rotation = 0

    def move(self, diff_coordinates):
        self.x += diff_coordinates[0]
        self.y += diff_coordinates[1]

    def rotate(self, can_rotate):
        new_rotation = (self.rotation + 1) % len(self.shapes)
        new_shape = self.shapes[new_rotation]

        if can_rotate(new_shape):
            self.rotation = new_rotation
            self.active_shape = new_shape
