from games.dodgeball import dodgeball
from games.reflex import reflex

from ui.menu import Menu


def main():

    while True:

        menu = Menu()

        game, duration = menu.run()

        if game == "quit":
            break

        elif game == "dodgeball":

            dodgeball.run()

        elif game == "reflex":

            reflex.run(duration=duration)


if __name__ == "__main__":
    main()