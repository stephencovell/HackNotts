import os


def main_menu() -> None:
    """
    Main menu for the program
    :return: None
    """
    # get project name from input
    project_name = input("Enter project name: ")
    project_exists: bool = check_for_project_folder(project_name)

    if project_exists:  # check if project already exists
        print("Project already exists")

        continue_with_project = "invalid"

        while continue_with_project not in ["y", "n"]:
            continue_with_project = input("Continue with project? (y/n): ").lower()

            if continue_with_project == "n":
                main_menu()

    else:
        create_new_project(project_name)


def create_new_project(project_name: str) -> None:
    """
    Creates a new project folder
    :param project_name: Name of the project
    :return: None
    """
    os.mkdir(project_name)




def check_for_project_folder(project_name: str) -> bool:
    """
    Checks if the project folder already exists
    :param project_name: Name of the project
    :return (bool): True if project already exists, False if not
    """

    if os.path.exists(project_name):
        return True
    else:
        return False


if __name__ == "__main__":
    main_menu()
