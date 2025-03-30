import glob
import logging

from seqeval.metrics.sequence_labeling import classification_report

import utils

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler('logs/eval.log', mode='a', encoding='utf-8')
        ]
    )

    gold_path = "data/test.conll"
    y_true = [[l for t, l in sent] for sent in utils.read_data(gold_path)]

    for pred_fp in glob.glob("output/*.conll"):
        logging.info(f"Evaluation of predictions in [{pred_fp}]")
        y_pred = [[l for t, l in sent] for sent in utils.read_data(pred_fp)]
        report = classification_report(y_true, y_pred, scheme="IOB2")
        logging.info("\n" + report + "\n")
