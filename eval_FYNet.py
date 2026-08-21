"""
Thin driver that enforces single-frequency input, then hands off to
eval_MFISNet_Fused.py (a single-frequency MFISNet_Fused run is
equivalent to FYNet).
"""
import logging

from eval_MFISNet_Fused import setup_args, main
from src.utils.logging_utils import FMT, TIMEFMT

if __name__ == "__main__":
    a = setup_args()

    if len(a.data_input_nus) != 1:
        raise ValueError(
            f"eval_FYNet.py requires exactly one frequency in --data_input_nus "
            f"(a single-frequency MFISNet_Fused run is what makes this FYNet); "
            f"received {a.data_input_nus}. For multiple frequencies, use "
            f"eval_MFISNet_Fused.py directly."
        )

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    logging.info(f"Received the following arguments: {a}")
    main(a)
