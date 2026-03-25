from manim import *
from config import BACKGROUND_COLOR


class TrainingResults(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        # ==========================================
        # 1. Title Screen
        # ==========================================
        title = Text("Results", font_size=48, weight=BOLD, color=TEAL)

        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title, shift=UP))

        # ==========================================
        # 2. Gameplay Placeholders (Generations)
        # ==========================================
        # We place the text at the absolute top to leave the whole center/bottom empty

        generations = [0, 25, 100, 600]
        current_label = None

        for gen in generations:
            new_label = Text(
                f"{gen} Generations", font_size=36, weight=BOLD, color=WHITE
            ).to_edge(UP, buff=0.5)

            if current_label is None:
                # For the very first label, just fade it in
                self.play(FadeIn(new_label, shift=DOWN))
            else:
                # For the rest, smoothly morph the old number into the new number
                self.play(ReplacementTransform(current_label, new_label))

            current_label = new_label

            # ---> IMPORTANT <---
            # This wait time is how long the text stays on screen.
            # Change '5' to however many seconds of gameplay footage you want to show for each!
            self.wait(5)

        # Final fade out
        self.play(FadeOut(current_label))
        self.wait(1)
