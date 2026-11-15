import cv2
import numpy as np


WIDTH = 1000
HEIGHT = 650

WINDOW_NAME = "Motion AI"

FONT = cv2.FONT_HERSHEY_SIMPLEX

BACKGROUND = (27, 21, 16)
WHITE = (255, 255, 255)
GRAY = (170, 160, 150)

RED = (70, 70, 245)
BLUE = (255, 180, 55)
GREEN = (80, 220, 80)

DODGE_PANEL = (45, 28, 35)
REFLEX_PANEL = (55, 40, 25)

MENU_TITLE = "MOTION AI"


class Menu:

    def __init__(self):
        self.screen = "main"

        self.mouse_x = -1
        self.mouse_y = -1

        self.selection = None
        self.buttons = []


    def draw_text(
        self,
        frame,
        text,
        x,
        y,
        scale=1.0,
        color=WHITE,
        thickness=2,
    ):
        cv2.putText(
            frame,
            text,
            (int(x), int(y)),
            FONT,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


    def draw_centered_text(
        self,
        frame,
        text,
        x,
        y,
        scale=1.0,
        color=WHITE,
        thickness=2,
    ):
        size, _ = cv2.getTextSize(
            text,
            FONT,
            scale,
            thickness,
        )

        self.draw_text(
            frame,
            text,
            x - size[0] // 2,
            y,
            scale,
            color,
            thickness,
        )


    def inside_button(self, bounds, x, y):
        x1, y1, x2, y2 = bounds

        return (
            x1 <= x <= x2
            and y1 <= y <= y2
        )


    def draw_button(
        self,
        frame,
        bounds,
        label,
        action,
        color,
    ):
        x1, y1, x2, y2 = bounds

        hovered = self.inside_button(
            bounds,
            self.mouse_x,
            self.mouse_y,
        )

        button_color = color

        if hovered:
            button_color = tuple(
                min(255, value + 35)
                for value in color
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            button_color,
            -1,
        )

        self.draw_centered_text(
            frame,
            label,
            (x1 + x2) // 2,
            (y1 + y2) // 2 + 8,
            scale=0.7,
        )

        self.buttons.append({
            "bounds": bounds,
            "action": action,
        })


    def draw_background(self):
        frame = np.zeros(
            (HEIGHT, WIDTH, 3),
            dtype=np.uint8,
        )

        frame[:] = BACKGROUND

        return frame


    def draw_dodgeball_icon(self, frame, x, y):
        # Motion lines behind the ball.
        for offset in [-16, 0, 16]:

            cv2.line(
                frame,
                (x - 105, y + offset),
                (x - 48, y + offset),
                RED,
                3,
                cv2.LINE_AA,
            )

        # Main ball.
        cv2.circle(
            frame,
            (x, y),
            40,
            RED,
            -1,
            cv2.LINE_AA,
        )

        # Ball highlight.
        cv2.circle(
            frame,
            (x - 12, y - 12),
            12,
            (160, 160, 255),
            -1,
            cv2.LINE_AA,
        )


    def draw_reflex_icon(self, frame, x, y):
        cv2.circle(
            frame,
            (x, y),
            45,
            BLUE,
            3,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            (x, y),
            28,
            GREEN,
            3,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            (x, y),
            6,
            WHITE,
            -1,
            cv2.LINE_AA,
        )


    def draw_game_card(
        self,
        frame,
        x,
        y,
        width,
        height,
        name,
        description,
        subtitle,
        color,
        panel_color,
        action,
        key,
    ):
        center_x = x + width // 2

        # Card background.
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            panel_color,
            -1,
        )

        # Card border.
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            2,
            cv2.LINE_AA,
        )

        # Game icon.
        if action == "dodgeball":

            self.draw_dodgeball_icon(
                frame,
                center_x,
                y + 75,
            )

        else:

            self.draw_reflex_icon(
                frame,
                center_x,
                y + 75,
            )

        self.draw_centered_text(
            frame,
            name,
            center_x,
            y + 165,
            scale=1.3,
        )

        self.draw_centered_text(
            frame,
            description,
            center_x,
            y + 205,
            scale=0.55,
            color=GRAY,
            thickness=1,
        )

        self.draw_centered_text(
            frame,
            subtitle,
            center_x,
            y + 235,
            scale=0.50,
            color=GRAY,
            thickness=1,
        )

        self.draw_button(
            frame,
            (
                x + 40,
                y + height - 75,
                x + width - 40,
                y + height - 25,
            ),
            f"PLAY  [{key}]",
            action,
            color,
        )


    def draw_main_menu(self):
        frame = self.draw_background()

        self.draw_centered_text(
            frame,
            MENU_TITLE,
            WIDTH // 2,
            90,
            scale=2.0,
            thickness=3,
        )

        self.draw_centered_text(
            frame,
            "YOUR BODY IS THE CONTROLLER",
            WIDTH // 2,
            130,
            scale=0.6,
            color=GRAY,
            thickness=1,
        )

        # DodgeAI card.
        self.draw_game_card(
            frame,
            x=55,
            y=180,
            width=425,
            height=350,
            name="DodgeAI",
            description="Dodge adaptive attacks",
            subtitle="Survive as long as possible",
            color=RED,
            panel_color=DODGE_PANEL,
            action="dodgeball",
            key=1,
        )

        # ReflexAI card.
        self.draw_game_card(
            frame,
            x=520,
            y=180,
            width=425,
            height=350,
            name="ReflexAI",
            description="Test your reaction speed",
            subtitle="Build combos and score points",
            color=BLUE,
            panel_color=REFLEX_PANEL,
            action="reflex",
            key=2,
        )

        self.draw_centered_text(
            frame,
            "Powered by Google Gemini + MediaPipe",
            WIDTH // 2,
            585,
            scale=0.55,
            color=GRAY,
            thickness=1,
        )

        self.draw_centered_text(
            frame,
            "Q / ESC - Quit",
            WIDTH // 2,
            620,
            scale=0.5,
            color=GRAY,
            thickness=1,
        )

        return frame


    def draw_duration_menu(self):
        frame = self.draw_background()

        self.draw_centered_text(
            frame,
            "REFLEX AI",
            WIDTH // 2,
            100,
            scale=1.8,
            color=BLUE,
            thickness=3,
        )

        self.draw_centered_text(
            frame,
            "SELECT GAME DURATION",
            WIDTH // 2,
            150,
            scale=0.8,
            color=GRAY,
        )

        durations = [
            (30, "QUICK"),
            (60, "STANDARD"),
            (90, "EXTENDED"),
        ]

        for i, (duration, label) in enumerate(durations):

            x = 55 + i * 305
            y = 225

            width = 280
            height = 250

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                REFLEX_PANEL,
                -1,
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                BLUE,
                2,
            )

            self.draw_centered_text(
                frame,
                f"{duration}s",
                x + width // 2,
                y + 90,
                scale=2.0,
                thickness=3,
            )

            self.draw_centered_text(
                frame,
                label,
                x + width // 2,
                y + 135,
                scale=0.7,
                color=GRAY,
            )

            self.draw_button(
                frame,
                (
                    x + 30,
                    y + 175,
                    x + width - 30,
                    y + 225,
                ),
                f"PLAY  [{i + 1}]",
                duration,
                BLUE,
            )

        self.draw_button(
            frame,
            (400, 540, 600, 590),
            "BACK",
            "back",
            (70, 70, 70),
        )

        return frame


    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):
        self.mouse_x = x
        self.mouse_y = y

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for button in self.buttons:

            if self.inside_button(
                button["bounds"],
                x,
                y,
            ):
                self.selection = button["action"]

                break


    def handle_action(self, action):

        if action == "quit":
            return ("quit", None)

        if action == "back":

            self.screen = "main"
            return None

        if self.screen == "main":

            if action == "dodgeball":
                return ("dodgeball", None)

            if action == "reflex":

                self.screen = "duration"
                return None

        elif self.screen == "duration":

            if action in [30, 60, 90]:
                return ("reflex", action)

        return None


    def handle_key(self, key):

        if key == ord("q"):
            return "quit"

        if key == 27:

            if self.screen == "duration":
                return "back"

            return "quit"

        if self.screen == "main":

            if key == ord("1"):
                return "dodgeball"

            if key == ord("2"):
                return "reflex"

        elif self.screen == "duration":

            if key == ord("1"):
                return 30

            if key == ord("2"):
                return 60

            if key == ord("3"):
                return 90

        return None


    def run(self):

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE,
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self.mouse_callback,
        )

        try:

            while True:

                self.buttons = []

                if self.screen == "main":
                    frame = self.draw_main_menu()

                else:
                    frame = self.draw_duration_menu()

                cv2.imshow(
                    WINDOW_NAME,
                    frame,
                )

                key = cv2.waitKey(16) & 0xFF

                action = self.selection
                self.selection = None

                if action is None:
                    action = self.handle_key(key)

                if action is not None:

                    result = self.handle_action(
                        action
                    )

                    if result is not None:
                        return result

                # Detect when the window is closed.
                try:

                    visible = cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )

                    if visible < 1:
                        return ("quit", None)

                except cv2.error:
                    return ("quit", None)

        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)