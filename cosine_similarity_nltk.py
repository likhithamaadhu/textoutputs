import os
from fuzzywuzzy import fuzz

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    words = [
        word.lower()
        for word in words
        if word.isalnum() and word.lower() not in stop_words
    ]
    return " ".join(words)


def calculate_cosine_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cosine_sim[0][0]


folder_paths = [
    "/home/likhithamaadhu/comparision/aws",
    "/home/likhithamaadhu/comparision/azure",
    "/home/likhithamaadhu/comparision/gcp",
    "/home/likhithamaadhu/comparision/microsoft",
    "/home/likhithamaadhu/comparision/original",
]


def fuzzy_compare(text1, text2):
    ratio = fuzz.token_sort_ratio(text1, text2)
    return ratio


similarity_results = {}

# each folder
for i in range(len(folder_paths)):
    folder1 = folder_paths[i]
    for j in range(i + 1, len(folder_paths)):
        folder2 = folder_paths[j]

        # first folder
        for file1 in os.listdir(folder1):
            file1_path = os.path.join(folder1, file1)

            # second folder
            for file2 in os.listdir(folder2):
                file2_path = os.path.join(folder2, file2)

                # Read and preprocess of data of files
                with open(file1_path, "r") as f1, open(file2_path, "r") as f2:
                    text1 = preprocess_text(f1.read())
                    text2 = preprocess_text(f2.read())
                    # text1_vector = get_embedding(
                    #     f1.read(), engine="text-embedding-ada-002"
                    # )
                    # text2_vector = get_embedding(
                    #     f2.read(), engine="text-embedding-ada-002"
                    # )

                    # similarity calculation
                    similarity = cosine_similarity(text1, text2)
                    ratio = fuzzy_compare(text1, text2)

                    key = f"{file1} - {file2}"
                    similarity_results[key] = (similarity, ratio)


# for key, similarity in similarity_results.items():
#     print(f"{key}: {similarity}")
# print(similarity_results)


output_file = "similarity_results_added.txt"
with open(output_file, "w") as file:
    for key, (similarity, ratio) in similarity_results.items():
        if similarity >= 0.3:
            file.write(f"{key}: {similarity},{ratio}\n")

print(f"Similarity results written to {output_file}")
