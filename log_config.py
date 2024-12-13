import logging
logging.basicConfig(level=logging.DEBUG)

# Tambahkan log setelah preprocessing
logging.debug(f"Preprocessed image shape: {preprocessed_image.shape}")

# Tambahkan log untuk hasil prediksi
logging.debug(f"Raw predictions: {predictions}")
logging.debug(f"Decoded texts: {decoded_texts}")
