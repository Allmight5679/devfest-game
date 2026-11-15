DEFAULT_GAME_DURATION = 60

GAME_DURATION_OPTIONS = [
    30,
    60,
    90,
]


def get_duration(score):
    if score < 5:
        return 3.0

    elif score < 10:
        return 2.5

    elif score < 20:
        return 2.0

    elif score < 30:
        return 1.5

    elif score < 40:
        return 1.2

    else:
        return 0.8