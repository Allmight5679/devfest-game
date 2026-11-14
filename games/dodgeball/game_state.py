from .collision import check_collision


STARTING_LIVES = 3
MESSAGE_DURATION = 1.0

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class GameState:

    def __init__(self):
        self.reset()


    def reset(self):
        self.ball = None

        self.score = 0
        self.lives = STARTING_LIVES
        self.game_over = False

        self.message = ""
        self.message_color = WHITE
        self.message_time = 0.0


    def update_ball(
        self,
        opponent,
        frame,
        points,
        valid_player,
        in_grace_period,
        delta_time,
    ):
        # Do not update gameplay after game over.
        if self.game_over:
            return

        if valid_player:

            # Spawn a new ball when needed.
            if self.ball is None:
                self.ball = opponent.spawn_ball(
                    frame,
                    points,
                    self.score,
                )

            else:
                self.ball.update(delta_time)

        elif in_grace_period:

            # Continue animation during brief
            # tracking interruptions.
            if self.ball is not None:
                self.ball.update(delta_time)


    def resolve_collision(
        self,
        mask,
        valid_player,
        current_time,
    ):
        # Only check collisions at impact.
        if (
            self.game_over
            or not valid_player
            or self.ball is None
            or not self.ball.finished
        ):
            return

        result = check_collision(self.ball, mask)

        # Wait if tracking is unavailable.
        if result is None:
            return

        if result:
            self.lives -= 1

            self.message = "HIT!"
            self.message_color = RED

        else:
            self.score += 1

            self.message = "DODGED!"
            self.message_color = GREEN

        self.message_time = current_time

        print(
            f"{self.message} "
            f"Score: {self.score}, "
            f"Lives: {self.lives}"
        )

        # Remove the resolved ball.
        self.ball = None

        # Check whether the game has ended.
        if self.lives <= 0:
            self.game_over = True

        return result


    def message_active(self, current_time):
        return (
            bool(self.message)
            and current_time - self.message_time < MESSAGE_DURATION
            and not self.game_over
        )