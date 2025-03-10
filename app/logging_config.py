import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # or INFO, WARNING, ERROR, CRITICAL
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )