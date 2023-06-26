import os
import functions_framework
import torch

from google.cloud import storage
from detect import handle_detect_language_request
from transformers import AutoTokenizer, BertForMaskedLM


# Global storage client
storage_client = storage.Client()
# Global model instance
model = None
# Global tokenizer instance
tokenizer = None


@functions_framework.http
def predict(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    Note:
        For more information on how Flask integrates with Cloud
        Functions, see the `Writing HTTP functions` page.
        <https://cloud.google.com/functions/docs/writing/http#http_frameworks>
    """
    params = request.get_json(silent=True)
    return handle_detect_language_request(params)


def download_model():
    bucket_name = os.environ["MODEL_BUCKET"]
    model_folder = os.environ["MODEL_FOLDER"]

    folder = "/tmp/local_model"
    if not os.path.exists(folder):
        os.makedirs(folder)

    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_folder)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_name = os.path.basename(blob.name)
        blob.download_to_filename(os.path.join(folder, file_name))


@functions_framework.http
def predict_external(request):
    global model
    global tokenizer

    if not model or not tokenizer:
        download_model()
        model = BertForMaskedLM.from_pretrained("/tmp/local_model")
        tokenizer = AutoTokenizer.from_pretrained("/tmp/local_model")

    params = request.get_json(silent=True)
    inputs = tokenizer(params["text"], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
        as_tuple=True
    )[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

    return tokenizer.decode(predicted_token_id)
