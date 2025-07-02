def show_main_menu():
    print("\n==============================")
    print("   FACE RECOGNITION SYSTEM")
    print("==============================")
    print("Choose a mode:")
    print("1. Face Recognition Only")
    print("2. Face Recognition + Anti-Spoofing")
    print("3. Exit")


def show_action_menu(mode):
    print(
        f"\nYou selected: {'Face Recognition Only' if mode == 'recognition' else 'Face Recognition + Anti-Spoofing'}"
    )
    print("Choose an action:")

    if mode == "recognition":
        print("1. Register face")
        print("2. Test")
        print("3. Go back")

    elif mode == "spoofing":
        print("1. Register face")
        print("2. Capture real/fake and train")
        print("3. Test")
        print("4. Go back")


def main():
    while True:
        show_main_menu()
        mode_input = input("Enter your choice (1-3): ").strip()

        if mode_input == "1":
            mode = "recognition"
        elif mode_input == "2":
            mode = "spoofing"
        elif mode_input == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid input. Try again.")
            continue

        while True:
            show_action_menu(mode)
            action_input = input("Enter your choice: ").strip()

            # Handle recognition mode
            if mode == "recognition":
                if action_input == "1":
                    action = "register"
                elif action_input == "2":
                    action = "test"
                elif action_input == "3":
                    break
                else:
                    print("Invalid input. Try again.")
                    continue

                from run_recognition import run_recognition

                run_recognition(action)

            # Handle spoofing mode
            elif mode == "spoofing":
                if action_input == "1":
                    action = "register"
                elif action_input == "2":
                    action = "capture"
                elif action_input == "3":
                    action = "test"
                elif action_input == "4":
                    break
                else:
                    print("Invalid input. Try again.")
                    continue

                from run_spoofing import run_spoofing

                run_spoofing(action)


if __name__ == "__main__":
    main()
