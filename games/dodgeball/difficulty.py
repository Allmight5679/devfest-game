# Determine ball travel duration based on score.

def get_duration(score):
    if score < 5:
        return 3.0

    elif score < 10:
        return 2.5

    elif score < 20:
        return 2.0

    elif score < 30:
        return 1.5

    elif score < 50:
        return 1.2

    else:
        return 0.8