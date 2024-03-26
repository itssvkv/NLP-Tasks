import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize


def get_user_text():
    """Prompts the user for text input."""
    return input("Please enter the text: ")


def display_menu():
    """Prints the menu options for tokenization."""
    print("\nOptions:")
    print("1. Print tokenized words")
    print("2. Print tokenized sentences")
    print("3. Print output using split function")


def get_user_choice():
    """Gets a valid choice from the user (1-3)."""
    while True:
        try:
            choice = int(input("\nEnter your choice: "))
            if 1 <= choice <= 3:
                return choice
            else:
                print("\nInvalid choice. Please enter a number between 1 and 3.")
        except ValueError:
            print("\nInvalid input. Please enter a number.")


def tokenize_text(text, choice):
    """Tokenizes the text based on user choice."""
    if choice == 1:
        return word_tokenize(text)
    elif choice == 2:
        return sent_tokenize(text)
    else:
        return text.split(), text.split(".")  # Return both words and sentences


text = get_user_text()
display_menu()
choice = get_user_choice()

results = tokenize_text(text, choice)
if isinstance(results, tuple):  # Check if both words and sentences are returned
    print(f"words = {results[0]}")
    print(f"sentences = {results[1]}")
else:
    print(results)
