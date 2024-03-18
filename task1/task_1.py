import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize, sent_tokenize

txt = input("Please, Enter the Text : ")
print(
    "\n Options : \n 1-print tokenized words \n 2-print tokenized sentences \n 3-print output using split function."
)
choice = int(input("\nEnter Your Choice : "))

while choice < 1 or choice > 3:
    print("\nPlease choose a valid option")
    choice = int(input("Enter Your Choice : "))


if choice == 1:
    tokenized_words = word_tokenize(txt)
    print(tokenized_words)
elif choice == 2:
    tokenized_sentences = sent_tokenize(txt)
    print(tokenized_sentences)
else:
    words = txt.split(" ")
    sentences = txt.split(".")
    print(f"words = {words}")
    print(f"sentences = {sentences}")
