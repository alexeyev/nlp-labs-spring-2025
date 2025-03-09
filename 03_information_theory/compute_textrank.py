from summa import keywords
from summa import summarizer

if __name__ == "__main__":

    with open("wikipedia.txt", "r", encoding="utf-8") as rf:
        text = rf.read()

    """
        TextRank -- старая штука, 2004 год
        
        В этой библиотеке используется стеммер Портера для языков
        "arabic", "danish", "dutch", "english", "finnish", "french",
        "german", "hungarian", "italian", "norwegian", "polish",
        "portuguese", "romanian", "russian", "spanish", "swedish"
        
        И реализация алгоритма не особенно ориентирована 
          на практическое применение, поэтому построим по небольшому тексту,
          иначе завалится с проблемами с памятью
    """

    kws = keywords.keywords(text=text,
                            ratio=0.1,
                            language="english",
                            split=True,
                            scores=True,
                            deaccent=False,
                            additional_stopwords=None)

    print("<Keywords>")

    for word, score in sorted(kws, key=lambda x: x[1], reverse=True)[:6]:
        print(f"{word}\t{score:.4f}")

    summaries = summarizer.summarize(text=text,
                                     ratio=0.1,
                                     language="english",
                                     split=True,
                                     scores=True,
                                     additional_stopwords=None)

    print("\n<Summary>")

    for summary, score in sorted(summaries, key=lambda x: x[1], reverse=True)[:6]:
        print(f"{score:.4f}\t{summary[:90]}...")
