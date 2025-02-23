import csv
import logging
import os
import shutil

from tqdm import tqdm
from whoosh import index
from whoosh.fields import Schema, ID, TEXT
from whoosh.index import Index
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.query import Query
from whoosh.searching import Searcher, Results

from engine import StoredIndexEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler('whoosh.log', mode='a', encoding='utf-8')
    ]
)


class WhooshEngine(StoredIndexEngine):

    def __init__(self, index_path="indexdir"):
        super().__init__(index_storage_path=index_path)

        # Задаём схему как в БД -- для индекса,
        #   "тяжёлые" объекты можно было бы хранить отдельно
        self.schema = Schema(
            docno=ID(stored=True, unique=True),  # уникальный ключ
            title=TEXT(stored=True),  # заголовок
            text=TEXT(stored=True)  # текст документа
        )

        self.ix: Index = None
        self.parser: QueryParser = None

        # Если папка с файлами индекса уже есть,
        #   попробуем загрузить из неё индекс
        if os.path.exists(self.ix_path):
            logging.info("Pre-built index found, loading.")
            try:
                self.ix: Index = index.open_dir(self.ix_path, schema=self.schema)
                logging.info(f"A total of {self.ix.doc_count()} documents in the index")
                self.parser: QueryParser = MultifieldParser(["title", "text"],
                                                            self.ix.schema)
            except Exception as e:
                logging.exception("Error upon reading the index, cleaning. Should rebuild.")
                self.ix: Index = None
                self.parser: QueryParser = None

                # Удаляем нечитаемый индекс
                shutil.rmtree(self.ix_path)
                logging.warning("Index directory removed successfully.")

    def index(self, csv_file="data/all-ru.csv"):
        """ Собственно индексирование """

        # Если нет, создаём, если есть -- пытаемся дополнить
        if not os.path.exists(self.ix_path):
            os.mkdir(self.ix_path)
            self.ix = index.create_in(self.ix_path, schema=self.schema)
        else:
            # if already indexed
            self.ix = index.open_dir(self.ix_path, schema=self.schema)

        # Создаём специальный пишущий объект
        writer = self.ix.writer()

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            # Пропускаем заголовок
            next(reader)

            for row in tqdm(reader, "lines indexed"):
                docno, title, text = row
                writer.update_document(
                    docno=docno.strip(),
                    title=title.strip(),
                    text=text.strip()
                )

        # Командуем, чтобы всё было точно записано на диск
        #   и доступно для поиска по индексу
        writer.commit()

        # Индекса в этот момент могло не быть, так что задаём парсер здесь
        self.parser: QueryParser = MultifieldParser(["title", "text"],
                                                    self.ix.schema)

    def search(self, query_string: str, limit: int = 10):

        if self.ix is None:
            raise Exception("Index has not been built.")

        searcher: Searcher = self.ix.searcher()
        query: Query = self.parser.parse(query_string)
        results: Results = searcher.search(query, limit=limit)

        logging.info(f"\nFound {len(results)} results:")
        logging.info(f"q:[{query_string}] "
                     f"hits:[{';'.join(str(hit['docno']) for hit in results)}]")

        return [{"id": hit["docno"],
                 "title": hit["title"],
                 "text": hit["text"],
                 "rank": hit.rank,
                 "score": hit.score,
                 "docnum": hit.docnum} for hit in results]


if __name__ == "__main__":
    """
        Задание для самостоятельной работы (без проверки)
        1)  прикрутите лемматизацию (pymorphy2, mystem, spacy, etc.)
        2)  скачайте словарь синонимов и сделайте некоторое подобие расширения запроса
            (а если уже умеете работать с эмбеддингами, можете использовать и их)
        3)  попробуйте сделать подготовку сниппетов (класс Highlighter)
        4)  возьмите коллекцию русских текстов побольше и проиндексируйте её
        5*) попробуйте сделать с короткими текстами векторный поиск на эмбеддингах 
            предложений и с векторным индексом (faiss, annoy, да много их)
    """
    CsvFile = "data/all-ru.csv"
    IndexDir = "indexdir"

    engine = WhooshEngine(IndexDir)
    engine.index(CsvFile)
    logging.info(f"Indexing complete! Index contains {engine.ix.doc_count()} documents.")

    hits = engine.search("мимо сферы")

    for hit in hits:
        title = hit["title"].replace("\n", " ")
        print(f'{hit["docnum"]:4d}: {hit["rank"]:2d})'
              f'\t[{hit["score"]:03.3f}]\t{title}')
